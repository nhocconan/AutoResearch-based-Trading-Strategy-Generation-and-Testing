#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot long/short with 4h trend filter and volume confirmation
# - Primary signal: Price reverses from Camarilla H3/L3 levels on 1h
# - Trend filter: 4h EMA(21) slope (must align with trade direction)
# - Volume filter: 1h volume > 1.3x 20-period average volume
# - Session filter: Trade only 08:00-20:00 UTC to avoid low-liquidity hours
# - Position size: 0.20 discrete level to minimize fee churn
# - Stoploss: Camarilla H4/L4 levels (strong break of pivot structure)
# - Target: 60-150 total trades over 4 years (15-37/year) using 1h for timing, 4h for direction

name = "1h_4h_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(21) for trend filter
    close_4h = df_4h['close'].values
    ema_21 = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_slope = ema_21 - np.roll(ema_21, 1)  # Positive = rising, Negative = falling
    ema_slope[0] = 0  # First value has no previous
    ema_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_slope)
    
    # Pre-compute 1h volume spike filter
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1h > (1.3 * avg_volume_20)
    
    # Pre-compute 1h ATR(14) for Camarilla calculation
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1h Camarilla levels (using previous bar's OHLC)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    hl_range = high_1h - low_1h
    camarilla_h4 = close_1h + 1.5 * hl_range
    camarilla_h3 = close_1h + 1.0 * hl_range
    camarilla_l3 = close_1h - 1.0 * hl_range
    camarilla_l4 = close_1h - 1.5 * hl_range
    
    # Shift levels to use previous bar's calculation (no look-ahead)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h4[0] = np.nan
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    camarilla_l4[0] = np.nan
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(ema_slope_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_filter[i]:
            if position != 0:  # Close position outside session
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reaches Camarilla L3 (mean reversion) OR breaks below L4 (stoploss)
            if close_1h[i] <= camarilla_l3[i] or close_1h[i] < camarilla_l4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price reaches Camarilla H3 (mean reversion) OR breaks above H4 (stoploss)
            if close_1h[i] >= camarilla_h3[i] or close_1h[i] > camarilla_h4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla reversals with trend and volume filters
            if vol_spike[i]:
                # Long: Price touches/below L3 AND 4h EMA rising
                if close_1h[i] <= camarilla_l3[i] and ema_slope_aligned[i] > 0:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: Price touches/above H3 AND 4h EMA falling
                elif close_1h[i] >= camarilla_h3[i] and ema_slope_aligned[i] < 0:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals