#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with weekly trend filter (1w EMA34) and volume confirmation
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance. Breakouts above R3 or below S3
# with volume confirmation indicate strong momentum. Weekly EMA34 filter ensures we only trade in the
# direction of the higher-timeframe trend, avoiding counter-trend whipsaws. This structure has proven
# effective on ETHUSDT and SOLUSDT in both bull and bear markets. Target 12-37 trades/year to minimize fee drag.

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and weekly trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # R3 = Close + 1.1 * (High - Low) / 2
    # S3 = Close - 1.1 * (High - Low) / 2
    df_1d = df_1d.copy()
    df_1d['R3'] = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2.0
    df_1d['S3'] = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2.0
    R3_vals = df_1d['R3'].values
    S3_vals = df_1d['S3'].values
    
    # Align Camarilla levels to 12h timeframe (completed prior day only)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_vals)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_vals)
    
    # Weekly trend filter: EMA34 on weekly close
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # volume MA20, weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        R3_val = R3_aligned[i]
        S3_val = S3_aligned[i]
        weekly_ema = ema_34_1w_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > weekly EMA34 (bullish trend) AND volume spike
            if price > R3_val and price > weekly_ema and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below S3 AND price < weekly EMA34 (bearish trend) AND volume spike
            elif price < S3_val and price < weekly_ema and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or breakdown below S3 (trend reversal)
            # ATR-based stoploss: 2.0 * ATR below entry (using 12h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or price breaks below S3 (trend reversal)
            if price < stop_loss or price < S3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or breakout above R3 (trend reversal)
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or price breaks above R3 (trend reversal)
            if price > stop_loss or price > R3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals