#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Uses 1d Camarilla pivot levels (L3, L4, H3, H4) from prior day for mean reversion entries
# - Trend filter: 1w EMA(21) - only long when price > EMA, short when price < EMA
# - Volume confirmation: 1d volume > 1.5x 20-period average to ensure breakout strength
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~10-25 trades/year (40-100 total over 4 years) per 1d strategy guidelines
# - Novelty: Combines Camarilla pivot mean reversion with weekly trend filter to avoid counter-trend trades
# - Works in both bull/bear: Camarilla captures reversals, weekly trend filter prevents fighting major trends

name = "1d_1w_camarilla_trend_volume_v1"
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
    # Align 1w EMA to 1d timeframe (completed 1w bar only)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 1d price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # Camarilla uses prior day's OHLC
    
    # Calculate 1d Camarilla pivot levels from prior day
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.0*(high - low)
    # L3 = close - 1.0*(high - low)
    # L4 = close - 1.5*(high - low)
    # Use prior day's values (shift by 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    # First value will be invalid (no prior day), handled by min_periods in rolling
    
    camarilla_h4 = prior_close + 1.5 * (prior_high - prior_low)
    camarilla_h3 = prior_close + 1.0 * (prior_high - prior_low)
    camarilla_l3 = prior_close - 1.0 * (prior_high - prior_low)
    camarilla_l4 = prior_close - 1.5 * (prior_high - prior_low)
    
    # Volume confirmation: 1d volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # 1d ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(20, n):  # Start after 20 bars for volume average
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_h3[i]) or
            np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_l4[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            np.isnan(ema_21_1w_aligned[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR breaks above H4 (take profit)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or high[i] >= camarilla_h4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR breaks below L4 (take profit)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or low[i] <= camarilla_l4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla pivot breakouts with volume confirmation and trend filter
            # Long: price breaks above H3 AND volume spike AND price > weekly EMA (uptrend)
            if high[i] >= camarilla_h3[i] and volume_spike[i] and close[i] > ema_21_1w_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below L3 AND volume spike AND price < weekly EMA (downtrend)
            elif low[i] <= camarilla_l3[i] and volume_spike[i] and close[i] < ema_21_1w_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals