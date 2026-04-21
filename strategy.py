#!/usr/bin/env python3
"""
1d_KeltnerChannel_Breakout_1wTrend_VolumeConfirm_ATRStop_v1
Hypothesis: Daily Keltner Channel (20,2.0) breakout with weekly EMA34 trend filter and volume spike confirmation. Designed for low trade frequency (~10-25/year) to minimize fee drag and improve generalization across bull/bear markets. Uses 1d primary timeframe with 1w HTF for trend context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # === 1w trend filter: 34-period EMA ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1w volume average (20-period) for spike detection ===
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w[np.isnan(vol_ma_1w)] = 1.0  # avoid division by zero
    vol_ratio_1w = volume_1w / vol_ma_1w
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    
    # === ATR for Keltner Channel and dynamic stoploss (14-period on 1d) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Keltner Channels (20-period, 2.0 multiplier on 1d) ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2.0 * atr_14
    lower_keltner = ema_20 - 2.0 * atr_14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ratio_1w_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1w = ema_34_1w_aligned[i]
        vol_spike = vol_ratio_1w_aligned[i]
        upper_kt = upper_keltner[i]
        lower_kt = lower_keltner[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner + volume spike > 2.0 + price above 1w EMA34
            if price_close > upper_kt and vol_spike > 2.0 and price_close > trend_1w:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below lower Keltner + volume spike > 2.0 + price below 1w EMA34
            elif price_close < lower_kt and vol_spike > 2.0 and price_close < trend_1w:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 2.5 * ATR below highest since entry
                if price_close < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.5 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KeltnerChannel_Breakout_1wTrend_VolumeConfirm_ATRStop_v1"
timeframe = "1d"
leverage = 1.0