#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + chop regime filter
# - Uses Kaufman's Adaptive Moving Average (KAMA) for trend direction
# - RSI(14) for momentum confirmation (long when RSI>50, short when RSI<50)
# - Choppiness Index (CHOP) to filter ranging markets (trade only when CHOP<38.2 = trending)
# - HTF: 1w EMA(34) as major trend filter (avoid counter-trend trades)
# - Signal size: 0.25 (discrete level to minimize fee churn)
# - Target: 15-25 trades/year on BTC/ETH, works in both bull (trend follow) and bear (avoid false signals via chop filter)
# - Proven pattern: KAMA+RSI+chop filter showed promise in DB for SOL; adapting for BTC/ETH with 1w HTF

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute volume data (not used in signals but available if needed)
    volume = prices['volume'].values
    
    # Pre-compute KAMA ( Kaufman's Adaptive Moving Average )
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array lengths: change is shorter by 10, volatility by 9
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility_padded > 0, change_padded / volatility_padded, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Pre-compute RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # First 14 gains (indices 1-13 in gain array)
    avg_loss[13] = np.mean(loss[1:14])  # First 14 losses
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute Choppiness Index (CHOP) - measures ranging vs trending
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # ATR(14) - sum of TR over 14 periods
    atr14 = np.full(n, np.nan)
    for i in range(14, n):
        atr14[i] = np.sum(tr[i-13:i+1])  # Sum of last 14 TR values
    
    # Highest high and lowest low over 14 periods
    hh14 = np.full(n, np.nan)
    ll14 = np.full(n, np.nan)
    for i in range(14, n):
        hh14[i] = np.max(high[i-13:i+1])
        ll14[i] = np.min(low[i-13:i+1])
    
    # CHOP = 100 * log10( sum(ATR14) / (HH14 - LL14) ) / log10(14)
    # Avoid division by zero
    range_14 = hh14 - ll14
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if range_14[i] > 0 and atr14[i] > 0:
            chop[i] = 100 * np.log10(atr14[i] / range_14[i]) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # Pre-compute 1w EMA(34) for HTF trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):  # Start after warmup period for all indicators
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine market regime: CHOP < 38.2 = trending (favorable for trend following)
        is_trending = chop[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long conditions:
            # 1. Price above KAMA (bullish trend)
            # 2. RSI > 50 (bullish momentum)
            # 3. Trending market (CHOP < 38.2)
            # 4. 1w EMA34 filter: price above weekly EMA (avoid buying in strong weekly downtrend)
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                is_trending and
                close[i] > ema34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions:
            # 1. Price below KAMA (bearish trend)
            # 2. RSI < 50 (bearish momentum)
            # 3. Trending market (CHOP < 38.2)
            # 4. 1w EMA34 filter: price below weekly EMA (avoid shorting in strong weekly uptrend)
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  is_trending and
                  close[i] < ema34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
                
        else:  # Have position - look for exit
            # Exit when trend weakens or reverses
            if position == 1:  # Long position
                # Exit if: price below KAMA OR RSI < 50 OR market becomes ranging (CHOP >= 38.2)
                if (close[i] < kama[i] or 
                    rsi[i] < 50 or 
                    chop[i] >= 38.2):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                # Exit if: price above KAMA OR RSI > 50 OR market becomes ranging (CHOP >= 38.2)
                if (close[i] > kama[i] or 
                    rsi[i] > 50 or 
                    chop[i] >= 38.2):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals