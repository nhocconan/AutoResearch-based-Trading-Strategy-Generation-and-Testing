#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d trend filter and volume confirmation
# - Long: Price touches Camarilla L3 support in 1d uptrend (close > EMA50) with volume spike
# - Short: Price touches Camarilla H3 resistance in 1d downtrend (close < EMA50) with volume spike
# - Volume filter: 12h volume > 1.5x 20-period average to confirm momentum
# - Position size: 0.30 discrete level for balanced risk/return
# - Stoploss: 2.0x ATR(20) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Trend filter ensures trades with higher probability; Camarilla levels provide institutional support/resistance

name = "12h_1d_camarilla_pivot_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 12h Camarilla levels from previous day
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate Camarilla levels using prior 12h bar's HLC
    # Shift by 1 to use completed prior bar
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]  # handle first bar
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    range_ = prev_high - prev_low
    camarilla_h3 = prev_close + range_ * 1.1 / 4
    camarilla_l3 = prev_close - range_ * 1.1 / 4
    
    # Pre-compute 12h volume spike filter
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * avg_volume_20)
    
    # Pre-compute 12h ATR(20) for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h3[i]) or
            np.isnan(camarilla_l3[i]) or np.isnan(vol_spike[i]) or np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price moves above L3 (mean reversion) OR stoploss hit
            if close_12h[i] > camarilla_l3[i] or close_12h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: Price moves below H3 (mean reversion) OR stoploss hit
            if close_12h[i] < camarilla_h3[i] or close_12h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Camarilla touch with trend and volume filters
            if vol_spike[i]:
                # Long: Price touches or crosses below L3 in uptrend
                if close_12h[i] <= camarilla_l3[i] and close_12h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.30
                # Short: Price touches or crosses above H3 in downtrend
                elif close_12h[i] >= camarilla_h3[i] and close_12h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.30
    
    return signals