#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour RSI divergence with weekly trend and volume confirmation
# Bullish divergence: price makes lower low, RSI makes higher low → long
# Bearish divergence: price makes higher high, RSI makes lower high → short
# Weekly trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation in the reversal
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "4h_RSIDivergence_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Find price and RSI extrema for divergence detection
    # Look for lows in price and RSI (for bullish divergence)
    price_low = pd.Series(low).rolling(window=5, center=True).min().values
    rsi_low = pd.Series(rsi).rolling(window=5, center=True).min().values
    # Look for highs in price and RSI (for bearish divergence)
    price_high = pd.Series(high).rolling(window=5, center=True).max().values
    rsi_high = pd.Series(rsi).rolling(window=5, center=True).max().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(price_low[i]) or 
            np.isnan(rsi_low[i]) or np.isnan(price_high[i]) or 
            np.isnan(rsi_high[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        pl = price_low[i]
        rl = rsi_low[i]
        ph = price_high[i]
        rh = rsi_high[i]
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if i >= 2:
                price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
                rsi_higher_low = rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]
                bullish_div = price_lower_low and rsi_higher_low
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
                rsi_lower_high = rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]
                bearish_div = price_higher_high and rsi_lower_high
                
                if bullish_div and ema50_1w_val > 0 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif bearish_div and ema50_1w_val < 0 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: bearish divergence or weekly trend turns down
            if i >= 2:
                price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
                rsi_lower_high = rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]
                bearish_div = price_higher_high and rsi_lower_high
                
                if bearish_div or ema50_1w_val < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish divergence or weekly trend turns up
            if i >= 2:
                price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
                rsi_higher_low = rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]
                bullish_div = price_lower_low and rsi_higher_low
                
                if bullish_div or ema50_1w_val > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals