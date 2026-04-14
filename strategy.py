# 1h_4h_1d_TrendBreakout_v1
# Hypothesis: Use 4h trend (price above/below 4h EMA 20) and 1d volatility regime (ATR ratio) to filter 1h breakouts.
# In bull markets: 4h EMA20 up + low volatility breakouts → long
# In bear markets: 4h EMA20 down + low volatility breakouts → short
# Volatility filter (current ATR < 1.5x 20-period ATR mean) avoids choppy markets and reduces false signals.
# Entry: 1h price breaks Donchian(10) channel in direction of 4h trend with volatility confirmation.
# Exit: Reverse signal or stop loss via ATR-based trailing stop (implemented as signal=0 when price moves against position).
# Target: 15-30 trades/year (60-120 total over 4 years) to stay within fee-efficient range.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Multi-timeframe indicators (computed once) ===
    # 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr2 = np.maximum(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]), tr1)
    atr_1d = pd.Series(tr2).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1h indicators for entry timing ===
    # Donchian channel (10 periods) for breakout
    donch_len = 10
    donch_upper = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().shift(1).values
    donch_lower = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().shift(1).values
    
    # 1h ATR for dynamic position sizing and stop loss
    tr1_h = np.maximum(high - low, np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr2_h = np.maximum(low - np.concatenate([[close[0]], close[:-1]]), tr1_h)
    atr_h = pd.Series(tr2_h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 8-20 UTC (reduces noise outside active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # Fixed 20% position size
    
    # Start after warmup period
    start = max(30, donch_len, 20)
    
    for i in range(start, n):
        # Skip if data not ready or outside session
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 1h ATR < 1.5x 20-period mean ATR (avoid choppy markets)
        atr_ma = pd.Series(atr_h).rolling(window=20, min_periods=1).mean().iloc[i] if i >= 20 else atr_h[i]
        vol_filter = atr_h[i] < 1.5 * atr_ma
        
        if position == 0:
            # Look for breakout in direction of 4h trend
            if vol_filter:
                # Long: price breaks above Donchian upper AND above 4h EMA20 (uptrend)
                if close[i] > donch_upper[i] and close[i] > ema_4h_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below Donchian lower AND below 4h EMA20 (downtrend)
                elif close[i] < donch_lower[i] and close[i] < ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Long position management
            # Exit if: price breaks below Donchian lower OR reverses below 4h EMA20
            if close[i] < donch_lower[i] or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Optional: trailing stop via ATR (2x ATR from highest high since entry)
            else:
                signals[i] = position_size
        elif position == -1:
            # Short position management
            # Exit if: price breaks above Donchian upper OR reverses above 4h EMA20
            if close[i] > donch_upper[i] or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_TrendBreakout_v1"
timeframe = "1h"
leverage = 1.0