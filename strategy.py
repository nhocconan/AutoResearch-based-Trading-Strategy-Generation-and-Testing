#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h Camarilla levels calculated from previous 4h bar's OHLC. Long when price breaks above R3 with
# 4h EMA50 uptrend and volume > 2.0x 20-bar average. Short when price breaks below S3 with
# 4h EMA50 downtrend and volume spike. Camarilla levels provide institutional support/resistance
# that work across bull/bear markets. Session filter (08-20 UTC) reduces noise trades.
# Target 15-37 trades/year via tight R3/S3 breakout conditions and session filter.
# Uses 1h primary timeframe with 4h HTF for signal direction, minimizing fee drag.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    hl_range = df_4h['high'] - df_4h['low']
    r3 = typical_price + hl_range * 1.1 / 4
    s3 = typical_price - hl_range * 1.1 / 4
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe (completed 4h bars only)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3.values)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # volume MA20 and EMA50 need sufficient history
    
    for i in range(start_idx, n):
        # Skip if not in trading session or any required data is NaN
        if not in_session[i] or (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
                                 np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema50_val = ema50_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND 4h EMA50 uptrend AND volume spike
            if price > r3_val and price > ema50_val and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short entry: price breaks below S3 AND 4h EMA50 downtrend AND volume spike
            elif price < s3_val and price < ema50_val and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price falls below S3 (reversal)
            # ATR-based stoploss: 2.0 * ATR below entry (using 1h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or price < S3 (reversal below support)
            if price < stop_loss or price < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on stoploss or price rises above R3 (reversal)
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or price > R3 (reversal above resistance)
            if price > stop_loss or price > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals