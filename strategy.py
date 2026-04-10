#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d trend filter (EMA50) + volume confirmation
# - Alligator: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
# - Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 (uptrend) AND volume > 1.8x 20-period average
# - Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 (downtrend) AND volume > 1.8x 20-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) to manage risk
# - Designed for 4h timeframe: targets 20-50 trades/year to avoid fee drag
# - Works in bull/bear markets: trend filter prevents counter-trend trades, Alligator captures trend strength

name = "4h_1d_alligator_volume_v1"
timeframe = "4h"
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
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 4h Williams Alligator components
    close_4h = prices['close'].values
    # Jaw: EMA13 with 8-bar shift
    jaw = pd.Series(close_4h).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    # Teeth: EMA8 with 5-bar shift
    teeth = pd.Series(close_4h).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    # Lips: EMA5 with 3-bar shift
    lips = pd.Series(close_4h).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.8 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks OR stoploss hit
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR stoploss hit
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator signals with trend and volume filters
            if vol_spike[i]:
                # Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 (uptrend)
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and close_4h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 (downtrend)
                elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close_4h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals