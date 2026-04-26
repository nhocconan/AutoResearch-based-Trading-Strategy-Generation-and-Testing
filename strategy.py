#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: Trade in direction of Kaufman Adaptive Moving Average (KAMA) on daily timeframe with RSI momentum confirmation and Choppiness Index regime filter. KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends. RSI (14) filters for momentum strength (>50 for long, <50 for short). Choppiness Index (14) > 61.8 avoids trading in strong trends where mean reversion fails. Designed for 1d timeframe to minimize trade frequency and fee drag, targeting 7-25 trades/year. Works in both bull (trend following) and bear (avoids false signals via chop filter) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (HTF = 1w as per experiment)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d (primary timeframe)
    # Note: KAMA requires close prices and ER (efficiency ratio)
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10))  # 10-period net change
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0).clip(0, 1)  # ER between 0 and 1
    
    # Smoothing constants: fastest = 2/(2+1), slowest = 2/(30+1)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14) on 1d
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when no loss
    
    # Calculate Choppiness Index (14) on 1d
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum()
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max()
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min()
    chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(atr_period)
    chop = chop.fillna(50)  # neutral when undefined
    
    # Get 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align 1d indicators to 1d timeframe (no alignment needed as primary TF is 1d)
    kama_aligned = kama  # already on 1d
    rsi_aligned = rsi.values  # already on 1d
    chop_aligned = chop.values  # already on 1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA calculation (need 10 for ER + 1 for init), RSI (14), Chop (14), EMA (50), Vol MA (20)
    start_idx = max(10, 14, 14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when market is choppy (range-bound) to avoid false breakouts
        # Chop > 61.8 indicates ranging market (good for mean reversion/breakout)
        # Chop <= 61.8 indicates trending market (avoid to reduce whipsaws)
        in_choppy_regime = chop_val > 61.8
        
        if position == 0:
            # Long: Price above KAMA (bullish bias) AND RSI > 50 (momentum) AND volume spike AND choppy regime
            long_signal = (close_val > kama_val) and (rsi_val > 50) and vol_spike and in_choppy_regime
            
            # Short: Price below KAMA (bearish bias) AND RSI < 50 (momentum) AND volume spike AND choppy regime
            short_signal = (close_val < kama_val) and (rsi_val < 50) and vol_spike and in_choppy_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price crosses below KAMA OR RSI drops below 40 (loss of momentum)
            if (close_val < kama_val) or (rsi_val < 40):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price crosses above KAMA OR RSI rises above 60 (loss of momentum)
            if (close_val > kama_val) or (rsi_val > 60):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0