#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + choppiness regime filter
# - KAMA(10,2,30) determines primary trend direction (long when price > KAMA, short when price < KAMA)
# - RSI(14) for momentum confirmation (long when RSI > 50, short when RSI < 50)
# - Choppiness Index(14) as regime filter: trade only when CHOP < 61.8 (trending market)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute KAMA(10,2,30)
    def kama_indicator(close, er_length=10, fast_sc=2, slow_sc=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < er_length:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'shape') else np.sum(np.abs(np.diff(close)))
        # Manual calculation for efficiency ratio
        er = np.full(n, np.nan)
        for i in range(er_length, n):
            price_change = np.abs(close[i] - close[i-er_length])
            price_volatility = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 1.0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama[er_length] = close[er_length]  # Seed with close
        for i in range(er_length+1, n):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = kama_indicator(close, 10, 2, 30)
    
    # Pre-compute RSI(14)
    def rsi_indicator(close, length=14):
        n = len(close)
        rsi = np.full(n, np.nan)
        if n < length + 1:
            return rsi
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # Initial average
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        
        # Wilder smoothing
        for i in range(length+1, n):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        # Calculate RSI
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100
        rsi[avg_gain == 0] = 0
        return rsi
    
    rsi = rsi_indicator(close, 14)
    
    # Pre-compute Choppiness Index(14)
    def chop_indicator(high, low, close, length=14):
        n = len(close)
        chop = np.full(n, np.nan)
        if n < length * 2:  # Need enough data for ATR calculation
            return chop
        
        # True Range
        tr = np.full(n, np.nan)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # ATR
        atr = np.full(n, np.nan)
        atr[length-1] = np.mean(tr[1:length])  # Simple average for first ATR
        for i in range(length, n):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        # Highest high and lowest low over period
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        for i in range(length-1, n):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        
        # Choppiness Index
        for i in range(length-1, n):
            if not np.isnan(atr[i]) and atr[i] > 0:
                sum_atr = np.sum(atr[i-length+1:i+1])
                if highest_high[i] - lowest_low[i] > 0:
                    chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(length)
                else:
                    chop[i] = 50  # Neutral when no range
            else:
                chop[i] = 50
        return chop
    
    chop = chop_indicator(high, low, close, 14)
    
    # Pre-compute 1w trend filter (optional: only trade with weekly trend)
    close_1w = df_1w['close'].values
    def sma_indicator(arr, length):
        n = len(arr)
        sma = np.full(n, np.nan)
        for i in range(length-1, n):
            sma[i] = np.mean(arr[i-length+1:i+1])
        return sma
    
    sma_50_1w = sma_indicator(close_1w, 50)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Choppiness regime filter: only trade in trending markets (CHOP < 61.8)
            if chop[i] < 61.8:
                # Long conditions: price > KAMA AND RSI > 50 AND weekly uptrend
                if close[i] > kama[i] and rsi[i] > 50 and weekly_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short conditions: price < KAMA AND RSI < 50 AND weekly downtrend
                elif close[i] < kama[i] and rsi[i] < 50 and weekly_downtrend:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Choppy market - stay flat
        else:  # Have position - look for exit
            # Exit conditions: reverse signal or choppy market
            exit_long = (position == 1 and (close[i] <= kama[i] or rsi[i] <= 50 or chop[i] >= 61.8))
            exit_short = (position == -1 and (close[i] >= kama[i] or rsi[i] >= 50 or chop[i] >= 61.8))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + choppiness regime filter
# - KAMA(10,2,30) determines primary trend direction (long when price > KAMA, short when price < KAMA)
# - RSI(14) for momentum confirmation (long when RSI > 50, short when RSI < 50)
# - Choppiness Index(14) as regime filter: trade only when CHOP < 61.8 (trending market)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute KAMA(10,2,30)
    def kama_indicator(close, er_length=10, fast_sc=2, slow_sc=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < er_length:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'shape') else np.sum(np.abs(np.diff(close)))
        # Manual calculation for efficiency ratio
        er = np.full(n, np.nan)
        for i in range(er_length, n):
            price_change = np.abs(close[i] - close[i-er_length])
            price_volatility = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 1.0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama[er_length] = close[er_length]  # Seed with close
        for i in range(er_length+1, n):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = kama_indicator(close, 10, 2, 30)
    
    # Pre-compute RSI(14)
    def rsi_indicator(close, length=14):
        n = len(close)
        rsi = np.full(n, np.nan)
        if n < length + 1:
            return rsi
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # Initial average
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        
        # Wilder smoothing
        for i in range(length+1, n):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        # Calculate RSI
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100
        rsi[avg_gain == 0] = 0
        return rsi
    
    rsi = rsi_indicator(close, 14)
    
    # Pre-compute Choppiness Index(14)
    def chop_indicator(high, low, close, length=14):
        n = len(close)
        chop = np.full(n, np.nan)
        if n < length * 2:  # Need enough data for ATR calculation
            return chop
        
        # True Range
        tr = np.full(n, np.nan)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # ATR
        atr = np.full(n, np.nan)
        atr[length-1] = np.mean(tr[1:length])  # Simple average for first ATR
        for i in range(length, n):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        # Highest high and lowest low over period
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        for i in range(length-1, n):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        
        # Choppiness Index
        for i in range(length-1, n):
            if not np.isnan(atr[i]) and atr[i] > 0:
                sum_atr = np.sum(atr[i-length+1:i+1])
                if highest_high[i] - lowest_low[i] > 0:
                    chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(length)
                else:
                    chop[i] = 50  # Neutral when no range
            else:
                chop[i] = 50
        return chop
    
    chop = chop_indicator(high, low, close, 14)
    
    # Pre-compute 1w trend filter (optional: only trade with weekly trend)
    close_1w = df_1w['close'].values
    def sma_indicator(arr, length):
        n = len(arr)
        sma = np.full(n, np.nan)
        for i in range(length-1, n):
            sma[i] = np.mean(arr[i-length+1:i+1])
        return sma
    
    sma_50_1w = sma_indicator(close_1w, 50)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Choppiness regime filter: only trade in trending markets (CHOP < 61.8)
            if chop[i] < 61.8:
                # Long conditions: price > KAMA AND RSI > 50 AND weekly uptrend
                if close[i] > kama[i] and rsi[i] > 50 and weekly_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short conditions: price < KAMA AND RSI < 50 AND weekly downtrend
                elif close[i] < kama[i] and rsi[i] < 50 and weekly_downtrend:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Choppy market - stay flat
        else:  # Have position - look for exit
            # Exit conditions: reverse signal or choppy market
            exit_long = (position == 1 and (close[i] <= kama[i] or rsi[i] <= 50 or chop[i] >= 61.8))
            exit_short = (position == -1 and (close[i] >= kama[i] or rsi[i] >= 50 or chop[i] >= 61.8))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals