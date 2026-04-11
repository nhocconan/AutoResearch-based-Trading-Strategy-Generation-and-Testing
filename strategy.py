#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Trend Reversal using 1d RSI divergence and 1w trend filter.
# Long when 1d RSI makes higher low while price makes lower low (bullish divergence) and 1w trend up.
# Short when 1d RSI makes lower high while price makes higher high (bearish divergence) and 1w trend down.
# Uses momentum divergence to detect exhaustion in trends, effective in both bull and bear markets.
# Target: 15-30 trades/year on 12h timeframe.

name = "12h_1d_1w_rsi_divergence_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate 1w EMA(40) for trend filter
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Track recent lows/highs for divergence detection
    lookback = 10  # Look back 10 periods for swing points
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_40_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Find recent swing low and high in price and RSI
        # Look back 'lookback' periods to find lowest low and highest high
        start_idx = max(0, i - lookback)
        end_idx = i
        
        # Price swing points
        price_low = np.min(low[start_idx:end_idx+1])
        price_high = np.max(high[start_idx:end_idx+1])
        price_low_idx = np.where(low[start_idx:end_idx+1] == price_low)[0][0] + start_idx
        price_high_idx = np.where(high[start_idx:end_idx+1] == price_high)[0][0] + start_idx
        
        # RSI at those points
        rsi_at_price_low = rsi_1d_aligned[price_low_idx] if not np.isnan(rsi_1d_aligned[price_low_idx]) else 50
        rsi_at_price_high = rsi_1d_aligned[price_high_idx] if not np.isnan(rsi_1d_aligned[price_high_idx]) else 50
        
        # Current RSI and price
        current_rsi = rsi_1d_aligned[i]
        current_price = close[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i > lookback:
            # Find previous swing low
            prev_start = max(0, i - lookback*2)
            prev_end = i - lookback
            if prev_end > prev_start:
                prev_price_low = np.min(low[prev_start:prev_end+1])
                prev_price_low_idx = np.where(low[prev_start:prev_end+1] == prev_price_low)[0][0] + prev_start
                prev_rsi_low = rsi_1d_aligned[prev_price_low_idx] if not np.isnan(rsi_1d_aligned[prev_price_low_idx]) else 50
                
                # Current low is lower than previous low, but RSI is higher
                if price_low < prev_price_low and rsi_at_price_low > prev_rsi_low:
                    bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i > lookback:
            # Find previous swing high
            prev_start = max(0, i - lookback*2)
            prev_end = i - lookback
            if prev_end > prev_start:
                prev_price_high = np.max(high[prev_start:prev_end+1])
                prev_price_high_idx = np.where(high[prev_start:prev_end+1] == prev_price_high)[0][0] + prev_start
                prev_rsi_high = rsi_1d_aligned[prev_price_high_idx] if not np.isnan(rsi_1d_aligned[prev_price_high_idx]) else 50
                
                # Current high is higher than previous high, but RSI is lower
                if price_high > prev_price_high and rsi_at_price_high < prev_rsi_high:
                    bearish_div = True
        
        # Determine 1w trend direction
        is_uptrend = close[i] > ema_40_1w_aligned[i]
        is_downtrend = close[i] < ema_40_1w_aligned[i]
        
        # Entry conditions: divergence with trend alignment
        # Long on bullish divergence in uptrend
        # Short on bearish divergence in downtrend
        long_signal = bullish_div and is_uptrend
        short_signal = bearish_div and is_downtrend
        
        # Exit conditions: opposite divergence or trend change
        exit_long = bearish_div or not is_uptrend
        exit_short = bullish_div or not is_downtrend
        
        # Priority: entry > exit > hold
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals