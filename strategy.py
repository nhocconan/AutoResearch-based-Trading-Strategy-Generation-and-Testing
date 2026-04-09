#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot long/short with 1w trend filter and volume confirmation
# - Uses 1d Camarilla pivot levels (H3/L3) for mean reversion entries in ranging markets
# - Trend filter: 1w EMA(21) to ensure alignment with weekly trend (avoid counter-trend in strong trends)
# - Volume confirmation: 1d volume > 1.5x 20-period average to ensure participation
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~10-25 trades/year (40-100 total over 4 years) per 1d strategy guidelines
# - Novelty: Combines Camarilla pivots with weekly trend filter to capture mean reversion in correct trend context
# - Works in both bull/bear: Camarilla pivots work in ranges, weekly trend filter avoids fighting strong trends

name = "1d_1w_camarilla_pivot_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    # Calculate 1w EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 1d price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # handle first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # 1d volume > 1.5x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # start after warmup for rolling calculations
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(close[i]) or
            np.isnan(high[i]) or
            np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price reaches Camarilla H3 or weekly trend turns bearish
            if high[i] >= camarilla_h3[i] or close[i] < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price reaches Camarilla L3 or weekly trend turns bullish
            if low[i] <= camarilla_l3[i] or close[i] > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla pivot touch with volume confirmation and trend alignment
            # Long: price touches/below Camarilla L3 AND volume spike AND price above weekly EMA (bullish trend)
            if low[i] <= camarilla_l3[i] and volume_spike[i] and close[i] > ema_21_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price touches/above Camarilla H3 AND volume spike AND price below weekly EMA (bearish trend)
            elif high[i] >= camarilla_h3[i] and volume_spike[i] and close[i] < ema_21_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals