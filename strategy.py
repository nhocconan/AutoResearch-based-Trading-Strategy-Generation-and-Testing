#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_1wTrend_VolumeConfirmation
Hypothesis: 1d timeframe strategy using Keltner Channel breakouts with 1-week EMA trend filter and volume confirmation (>1.5x 20-bar MA). Long when price breaks above upper KC in bullish 1w trend + volume spike; short when price breaks below lower KC in bearish 1w trend + volume spike. Exits on opposite KC touch or ATR-based stop (2.0x ATR). Designed for low trade frequency (15-25/year) to minimize fee drag while capturing structured breakouts in both bull and bear markets via trend alignment. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # 1w EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Previous day's OHLC for Keltner Channel (20-period, 2.0 ATR multiplier)
    # We need 20-day ATR of the 1d timeframe, but since we're on 1d chart,
    # we can calculate directly from daily data using HTF helper concept
    # However, we're on 1d timeframe, so we calculate KC from current 1d data
    # But to avoid look-ahead, we use previous day's data for KC calculation
    
    # For 1d timeframe, we need to calculate KC based on 20-period lookback
    # Since we're generating signals for 1d bars, we calculate KC from prior 20 days
    # We'll use rolling window on the 1d data itself (no HTF needed for KC calc)
    # But we must ensure we don't use current bar's data in KC calculation
    
    # Calculate 20-period EMA of close (for KC middle line)
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(20) for KC width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: Upper = EMA20 + 2.0*ATR20, Lower = EMA20 - 2.0*ATR20
    kc_upper = ema_20 + 2.0 * atr_20
    kc_lower = ema_20 - 2.0 * atr_20
    
    # Volume confirmation: volume > 1.5x 20-period EMA of volume
    volume_s = pd.Series(volume)
    vol_ema_20 = volume_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    # ATR for stoploss (using same ATR20 as above)
    atr_val = atr_20  # Reuse ATR20 for stoploss
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for EMA, 20 for ATR, 20 for volume EMA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(atr_val[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        kc_upper_val = kc_upper[i]
        kc_lower_val = kc_lower[i]
        ema_21_1w_val = ema_21_1w_aligned[i]
        vol_spike = volume_spike[i]
        atr_stop = atr_val[i]
        
        # Determine 1w trend: bullish if price > EMA21, bearish if price < EMA21
        bullish_1w = close_val > ema_21_1w_val
        bearish_1w = close_val < ema_21_1w_val
        
        # Entry conditions: breakout of KC in trend direction with volume spike
        long_entry = (close_val > kc_upper_val) and bullish_1w and vol_spike
        short_entry = (close_val < kc_lower_val) and bearish_1w and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -base_size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - check exit conditions
            exit_signal = False
            # Exit 1: price touches lower KC (mean reversion)
            if close_val < kc_lower_val:
                exit_signal = True
            # Exit 2: ATR-based stoploss (2.0x ATR below entry)
            elif close_val < entry_price - 2.0 * atr_stop:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - check exit conditions
            exit_signal = False
            # Exit 1: price touches upper KC (mean reversion)
            if close_val > kc_upper_val:
                exit_signal = True
            # Exit 2: ATR-based stoploss (2.0x ATR above entry)
            elif close_val > entry_price + 2.0 * atr_stop:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Keltner_Channel_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0