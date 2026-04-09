#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume and ADX trend filter
# - Primary signal: 4h price breaks above/below 1d Camarilla H3/L3 levels
# - Volume confirmation: 1d volume > 20-period EMA volume (institutional participation)
# - Trend filter: 1d ADX > 25 ensures we only trade in trending markets
# - Direction: In trending markets (ADX>25), breakouts must align with EMA20 trend
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla captures institutional levels, ADX filters chop

name = "4h_1d_camarilla_vol_adx_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0] if len(tr) > 0 else 0
    
    # 1d ADX calculation (Wilder's smoothing)
    period_adx = 14
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period_adx - 1) + tr[i]) / period_adx
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period_adx, adjust=False).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period_adx, adjust=False).mean().values / atr)
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/period_adx, adjust=False).mean().values
    
    # 1d EMA20 for trend direction
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d volume > 20-period EMA volume (institutional participation)
    volume_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_regime = volume_1d > volume_ema_20
    
    # 1d Camarilla pivot levels (based on prior day's range)
    camarilla_h3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align all 1d indicators to 4h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR trend weakens (ADX<25) OR trend reverses
            if close[i] <= camarilla_l3_aligned[i] or adx_aligned[i] < 25 or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR trend weakens (ADX<25) OR trend reverses
            if close[i] >= camarilla_h3_aligned[i] or adx_aligned[i] < 25 or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level breakout with volume and trend confirmation
            # Long: price breaks above Camarilla H3 AND volume regime AND ADX>25 AND above EMA20
            if (high[i] >= camarilla_h3_aligned[i] and 
                volume_regime_aligned[i] and 
                adx_aligned[i] > 25 and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Camarilla L3 AND volume regime AND ADX>25 AND below EMA20
            elif (low[i] <= camarilla_l3_aligned[i] and 
                  volume_regime_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals