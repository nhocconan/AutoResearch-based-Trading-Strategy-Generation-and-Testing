#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_Keltner_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # === 12h Keltner Channel ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 20-period EMA of typical price
    tp_12h = (high_12h + low_12h + close_12h) / 3
    ema_tp = pd.Series(tp_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # ATR(20)
    tr1_12h = np.abs(high_12h[1:] - low_12h[1:])
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Keltner Bands
    keltner_upper = ema_tp + 2.0 * atr_12h
    keltner_lower = ema_tp - 2.0 * atr_12h
    
    # Align to 4h
    keltner_upper_4h = align_htf_to_ltf(prices, df_12h, keltner_upper)
    keltner_lower_4h = align_htf_to_ltf(prices, df_12h, keltner_lower)
    ema_tp_4h = align_htf_to_ltf(prices, df_12h, ema_tp)
    
    # === 1d Volatility Regime (Keltner Width Percentile) ===
    tp_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    ema_tp_1d = pd.Series(tp_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    tr1_1d = np.abs(df_1d['high'][1:] - df_1d['low'][1:])
    tr2_1d = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3_1d = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    keltner_width_1d = (ema_tp_1d + 2.0 * atr_1d) - (ema_tp_1d - 2.0 * atr_1d)  # = 4 * atr_1d
    # Use 50-period percentile of width
    width_series = pd.Series(keltner_width_1d)
    width_rank = width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    # Normal regime: width between 30th and 70th percentile
    keltner_width_1d_aligned = align_htf_to_ltf(prices, df_1d, keltner_width_1d)
    width_rank_aligned = align_htf_to_ltf(prices, df_1d, width_rank)
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        close_val = prices['close'].iloc[i]
        up_val = keltner_upper_4h[i]
        low_val = keltner_lower_4h[i]
        ema_val = ema_tp_4h[i]
        vol_ratio_val = vol_ratio[i]
        width_rank_val = width_rank_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(up_val) or np.isnan(low_val) or 
            np.isnan(ema_val) or np.isnan(vol_ratio_val) or np.isnan(width_rank_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in normal volatility (avoid extreme volatility)
        in_normal_vol = (width_rank_val > 0.3) and (width_rank_val < 0.7)
        
        if position == 0 and in_normal_vol:
            # Long breakout above upper Keltner with volume
            if close_val > up_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short breakdown below lower Keltner with volume
            elif close_val < low_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle or breaks below lower band
            if close_val < ema_val or close_val < low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle or breaks above upper band
            if close_val > ema_val or close_val > up_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals