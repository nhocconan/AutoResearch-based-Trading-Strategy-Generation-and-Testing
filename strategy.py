#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 1d RSI divergence for mean reversion entries
# - Uses 12h HTF for Supertrend(ATR=10, mult=3.0): determines primary trend direction
# - Uses 1d HTF for RSI(14): looks for bullish/bearish divergence with price to catch reversals
# - In uptrend (price > Supertrend): look for long entries when bullish RSI divergence occurs
# - In downtrend (price < Supertrend): look for short entries when bearish RSI divergence occurs
# - Volume confirmation: current 6h volume > 1.5x 20-period average to filter low-quality signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Supertrend and RSI divergence combination works in both bull and bear markets by trading with the trend but picking optimal entry points during pullbacks

name = "6h_12h_1d_supertrend_rsi_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_basic = hl2 + (3.0 * atr_10)
    lower_basic = hl2 - (3.0 * atr_10)
    
    # Final Upper and Lower Bands
    final_upper = np.zeros_like(upper_basic)
    final_lower = np.zeros_like(lower_basic)
    
    for i in range(len(close_12h)):
        if np.isnan(atr_10[i]) or i == 0:
            final_upper[i] = upper_basic[i]
            final_lower[i] = lower_basic[i]
        else:
            if upper_basic[i] < final_upper[i-1] or close_12h[i-1] > final_upper[i-1]:
                final_upper[i] = upper_basic[i]
            else:
                final_upper[i] = final_upper[i-1]
                
            if lower_basic[i] > final_lower[i-1] or close_12h[i-1] < final_lower[i-1]:
                final_lower[i] = lower_basic[i]
            else:
                final_lower[i] = final_lower[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if np.isnan(atr_10[i]) or i == 0:
            supertrend[i] = np.nan
        elif supertrend[i-1] == final_upper[i-1]:
            if close_12h[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = final_lower[i]
        else:
            if close_12h[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d RSI divergence (bullish and bearish)
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    rsi_bullish_div = np.zeros_like(close_1d, dtype=bool)
    rsi_bearish_div = np.zeros_like(close_1d, dtype=bool)
    
    lookback = 5  # Look back 5 periods for divergence
    for i in range(lookback, len(close_1d)):
        if np.isnan(rsi[i]) or np.isnan(close_1d[i]):
            continue
            
        # Bullish divergence: lower price low, higher RSI low
        price_lower_low = close_1d[i] < close_1d[i-lookback:i].min()
        rsi_higher_low = rsi[i] > rsi[i-lookback:i][rsi[i-lookback:i] > 0].min() if np.any(rsi[i-lookback:i] > 0) else False
        rsi_bullish_div[i] = price_lower_low and rsi_higher_low
        
        # Bearish divergence: higher price high, lower RSI high
        price_higher_high = close_1d[i] > close_1d[i-lookback:i].max()
        rsi_lower_high = rsi[i] < rsi[i-lookback:i].max() if not np.all(np.isnan(rsi[i-lookback:i])) else False
        rsi_bearish_div[i] = price_higher_high and rsi_lower_high
    
    # Align all HTF data to 6h timeframe (wait for completed HTF bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    rsi_bullish_div_aligned = align_htf_to_ltf(prices, df_1d, rsi_bullish_div.astype(float))
    rsi_bearish_div_aligned = align_htf_to_ltf(prices, df_1d, rsi_bearish_div.astype(float))
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(rsi_bullish_div_aligned[i]) or np.isnan(rsi_bearish_div_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend determination: price above/below Supertrend
        bullish_trend = close[i] > supertrend_aligned[i]
        bearish_trend = close[i] < supertrend_aligned[i]
        
        # RSI divergence signals
        bullish_divergence = rsi_bullish_div_aligned[i] > 0.5
        bearish_divergence = rsi_bearish_div_aligned[i] > 0.5
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend:
                # In bullish trend: exit when bearish divergence or trend changes to bearish
                if bearish_divergence or bearish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Not in bullish trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_trend:
                # In bearish trend: exit when bullish divergence or trend changes to bullish
                if bullish_divergence or bullish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Not in bearish trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend and RSI divergence
            if volume_confirmed:
                if bullish_trend and bullish_divergence:
                    # In bullish trend, bullish divergence: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and bearish_divergence:
                    # In bearish trend, bearish divergence: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals