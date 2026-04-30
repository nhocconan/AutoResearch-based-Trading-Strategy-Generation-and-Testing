#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Williams Alligator (Jaw/Teeth/Lips) with volume confirmation and 1w EMA(34) trend filter
# Williams Alligator identifies trend absence (all lines intertwined) vs presence (lines diverged in order).
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
# 1w EMA(34) filter ensures trades align with weekly trend, reducing false signals in choppy markets.
# Volume confirmation (1.5x 20-period average) filters low-conviction breakouts.
# Designed for low trade frequency (~10-25/year on 1d) to minimize fee drag and improve bear market performance.

name = "1d_WilliamsAlligator_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price (typical price) with specific periods
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    # SMMA = smoothed moving average (EMA-like but different smoothing)
    # We'll approximate with EMA for simplicity and performance
    typical_price = (high + low + close) / 3.0
    
    # Jaw (13, 8)
    jaw = pd.Series(typical_price).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift 8 bars forward
    
    # Teeth (8, 5)
    teeth = pd.Series(typical_price).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift 5 bars forward
    
    # Lips (5, 3)
    lips = pd.Series(typical_price).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips.shift(3)  # shift 3 bars forward
    
    # Calculate 1w EMA(34) for trend filter
    close_1w_s = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 25  # warmup for Alligator (max shift 8 + max period 13)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_lips = lips.iloc[i] if not np.isnan(lips.iloc[i]) else 0
        curr_teeth = teeth.iloc[i] if not np.isnan(teeth.iloc[i]) else 0
        curr_jaw = jaw.iloc[i] if not np.isnan(jaw.iloc[i]) else 0
        curr_ema = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and Alligator alignment
            if volume_spike:
                # Bullish entry: Lips > Teeth > Jaw (bullish alignment) with weekly uptrend
                if curr_lips > curr_teeth > curr_jaw and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Lips < Teeth < Jaw (bearish alignment) with weekly downtrend
                elif curr_lips < curr_teeth < curr_jaw and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR Alligator turns bearish (Lips < Jaw)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_lips < curr_jaw:  # Alligator sleeping/bearish
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5 * ATR above entry (trailing profit)
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR Alligator turns bullish (Lips > Jaw)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_lips > curr_jaw:  # Alligator sleeping/bullish
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5 * ATR below entry (trailing profit)
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals