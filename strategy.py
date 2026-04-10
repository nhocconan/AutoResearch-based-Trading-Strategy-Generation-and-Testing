#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h Camarilla pivot + volume confirmation + session filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 4h
# - Trend filter: 1d EMA(50) slope confirms institutional trend direction
# - Volume filter: 1h volume > 1.5x 20-period average for momentum confirmation
# - Session filter: Only trade 08-20 UTC to avoid low-liquidity hours
# - Position size: 0.20 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 1h
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Camarilla pivots adapt to volatility; volume + trend filters avoid false breakouts

name = "1h_4h_1d_camarilla_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h Camarilla levels (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot and ranges using previous bar (shifted by 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]  # First bar uses current values
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    H4 = pivot + (range_hl * 1.1 / 2)
    L4 = pivot - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3)
    H4_aligned = align_htf_to_ltf(prices, df_4h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_4h, L4)
    
    # Pre-compute 1d EMA(50) slope for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_slope = ema_50 - np.roll(ema_50, 1)
    ema_slope[0] = 0
    ema_slope_pos = ema_slope > 0  # Uptrend
    ema_slope_neg = ema_slope < 0  # Downtrend
    ema_slope_pos_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_pos)
    ema_slope_neg_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_neg)
    
    # Pre-compute 1h volume spike filter
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1h > (1.5 * avg_volume_20)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_slope_pos_aligned[i]) or np.isnan(ema_slope_neg_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below L3 OR stoploss hit
            if close_1h[i] < L3_aligned[i] or close_1h[i] < entry_price - 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price crosses above H3 OR stoploss hit
            if close_1h[i] > H3_aligned[i] or close_1h[i] > entry_price + 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for breakout with trend and volume filters (only in session)
            if vol_spike[i]:
                # Long: Price breaks above H3 in uptrend
                if close_1h[i] > H3_aligned[i] and ema_slope_pos_aligned[i]:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: Price breaks below L3 in downtrend
                elif close_1h[i] < L3_aligned[i] and ema_slope_neg_aligned[i]:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals