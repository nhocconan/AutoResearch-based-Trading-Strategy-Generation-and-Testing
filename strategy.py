#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d trend and volume confirmation
# - Uses 4h EMA50 and 1d EMA200 for trend filter (avoid counter-trend trades)
# - Long: price breaks above H3 Camarilla pivot (4h) AND price > 4h EMA50 AND price > 1d EMA200 AND volume > 1.5x 20-period average
# - Short: price breaks below L3 Camarilla pivot (4h) AND price < 4h EMA50 AND price < 1d EMA200 AND volume > 1.5x 20-period average
# - Uses discrete position sizing (0.20) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) to manage risk
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Designed for 1h timeframe: targets 15-37 trades/year to avoid fee drag
# - Works in bull/bear markets: dual timeframe trend filter prevents counter-trend trades, Camarilla breakout captures momentum

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 4h Camarilla pivots (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = df_4h['close'].shift(1).values  # previous 4h close
    # Camarilla levels: H3 = close + (high-low)*1.1/4, L3 = close - (high-low)*1.1/4
    camarilla_h3 = close_4h_prev + (high_4h - low_4h) * 1.1 / 4
    camarilla_l3 = close_4h_prev - (high_4h - low_4h) * 1.1 / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
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
    
    # Pre-compute 1h volume confirmation
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1h > (1.5 * avg_volume_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_spike[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 camarilla OR stoploss hit
            if close_1h[i] < camarilla_l3_aligned[i] or close_1h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above H3 camarilla OR stoploss hit
            if close_1h[i] > camarilla_h3_aligned[i] or close_1h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above H3 camarilla AND price > 4h EMA50 AND price > 1d EMA200
                if close_1h[i] > camarilla_h3_aligned[i] and close_1h[i] > ema_50_4h_aligned[i] and close_1h[i] > ema_200_1d_aligned[i]:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: price breaks below L3 camarilla AND price < 4h EMA50 AND price < 1d EMA200
                elif close_1h[i] < camarilla_l3_aligned[i] and close_1h[i] < ema_50_4h_aligned[i] and close_1h[i] < ema_200_1d_aligned[i]:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals