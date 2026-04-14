# Your trading strategy code here
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy combining 4h ADX for trend strength and 1d RSI for mean reversion
# Uses 4h ADX > 25 to identify trending markets and 1d RSI < 30 or > 70 for entry signals
# In trending markets, buys pullbacks (RSI < 30) and sells rallies (RSI > 70)
# Works in both bull and bear markets: trend filter ensures we trade with the dominant trend,
# while RSI provides mean-reversion entries within the trend
# Uses session filter (08-20 UTC) to avoid low-liquidity periods and reduce false signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for ADX
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h ADX (14 periods)
    adx_len = 14
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values with min_periods
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Load 1d data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI (14 periods)
    rsi_len = 14
    delta = np.diff(df_1d['close'].values, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Smoothed gains and losses
    avg_gain = pd.Series(gain).rolling(window=rsi_len, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_len, min_periods=rsi_len).mean().values
    
    # Relative Strength
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle division by zero
    
    # Align RSI to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(100, adx_len + rsi_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Mean reversion signals from RSI
        oversold = rsi_aligned[i] < 30
        overbought = rsi_aligned[i] > 70
        
        if position == 0:
            # Enter long: trending + oversold (pullback in uptrend)
            if trending and oversold:
                # Additional filter: price above 20-period SMA for uptrend confirmation
                sma_20 = pd.Series(close[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
                if price > sma_20:
                    position = 1
                    signals[i] = position_size
            # Enter short: trending + overbought (pullback in downtrend)
            elif trending and overbought:
                # Additional filter: price below 20-period SMA for downtrend confirmation
                sma_20 = pd.Series(close[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
                if price < sma_20:
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion complete) or ADX drops
            if rsi_aligned[i] > 50 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion complete) or ADX drops
            if rsi_aligned[i] < 50 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hADX_1dRSI_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0