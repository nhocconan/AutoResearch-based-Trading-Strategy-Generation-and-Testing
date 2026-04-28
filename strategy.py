#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses 12h Camarilla levels calculated from previous 12h candle's OHLC. Long when price breaks above R1 with
# 12h EMA50 uptrend and volume > 1.8x 30-bar average. Short when price breaks below S1 with
# 12h EMA50 downtrend and volume spike. Camarilla levels provide institutional support/resistance
# that work across bull/bear markets. Target 20-50 trades/year via tight R1/S1 breakout conditions.
# Uses 4h primary timeframe as recommended for optimal balance of signal quality and trade frequency.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h candle's OHLC
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    hl_range = df_12h['high'] - df_12h['low']
    r1 = typical_price + hl_range * 1.1 / 12
    s1 = typical_price - hl_range * 1.1 / 12
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe (completed 12h levels only)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1.values)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: >1.8x 30-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 1.8 * volume_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 50)  # volume MA30 and EMA50 need sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_ma_30[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R1 AND 12h EMA50 uptrend AND volume spike
            if price > r1_val and price > ema50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below S1 AND 12h EMA50 downtrend AND volume spike
            elif price < s1_val and price < ema50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price falls below S1 (reversal)
            # ATR-based stoploss: 2.0 * ATR below entry (using 4h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or price < S1 (reversal below support)
            if price < stop_loss or price < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or price rises above R1 (reversal)
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or price > R1 (reversal above resistance)
            if price > stop_loss or price > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals