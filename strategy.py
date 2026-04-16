#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend with 1d volume regime filter and ATR-based position sizing.
# Long when Supertrend turns bullish AND 1d volume > 1.5x 20-period average.
# Short when Supertrend turns bearish AND 1d volume > 1.5x 20-period average.
# Position size scales with volatility: 0.30 when ATR(14) < 1.5*ATR(50), else 0.15.
# Uses discrete position sizes (0.0, ±0.15, ±0.30) to minimize fee churn.
# Supertrend captures trend, volume filter ensures participation, ATR scaling manages risk.
# Works in both bull and bear markets by following the trend with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Supertrend (ATR=10, mult=3.0) ===
    # True Range
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    final_upper[-1] = upper_band[-1]
    final_lower[-1] = lower_band[-1]
    
    for i in range(n-2, -1, -1):
        final_upper[i] = upper_band[i] if (close[i] <= final_upper[i+1] or final_upper[i+1] == 0) else final_upper[i+1]
        final_lower[i] = lower_band[i] if (close[i] >= final_lower[i+1] or final_lower[i+1] == 0) else final_lower[i+1]
    
    # Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    supertrend[-1] = final_lower[-1]
    direction[-1] = 1
    
    for i in range(n-2, -1, -1):
        if close[i] <= final_upper[i]:
            direction[i] = -1
            supertrend[i] = final_upper[i]
        elif close[i] >= final_lower[i]:
            direction[i] = 1
            supertrend[i] = final_lower[i]
        else:
            direction[i] = direction[i+1]
            supertrend[i] = supertrend[i+1] if direction[i] == 1 else final_upper[i]
    
    # Supertrend trend change signals
    supertrend_prev = np.roll(supertrend, 1)
    supertrend_prev[0] = supertrend[0]
    trend_change = (supertrend != supertrend_prev) & (np.arange(n) > 0)
    bullish_signal = trend_change & (direction == 1)
    bearish_signal = trend_change & (direction == -1)
    
    # === 1d Indicators: Volume Regime Filter ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === ATR for position sizing ===
    atr_14 = atr  # Already calculated ATR(10), but we need ATR(14) for comparison
    # Recalculate ATR(14) for volatility regime
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    vol_regime_low = atr_ratio < 1.5  # Low volatility regime
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(supertrend[i]) or np.isnan(supertrend_prev[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr_ratio[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # === EXIT LOGIC: Reverse on opposite Supertrend signal ===
        if position == 1 and bearish_signal[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and bullish_signal[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bullish Supertrend turn AND volume spike
            if bullish_signal[i] and volume_spike[i]:
                # Position size: 0.30 in low vol, 0.15 in high vol
                signals[i] = 0.30 if vol_regime_low[i] else 0.15
                position = 1
            
            # SHORT: Bearish Supertrend turn AND volume spike
            elif bearish_signal[i] and volume_spike[i]:
                # Position size: 0.30 in low vol, 0.15 in high vol
                signals[i] = -0.30 if vol_regime_low[i] else -0.15
                position = -1
        
        else:
            signals[i] = position * (0.30 if vol_regime_low[i] else 0.15)
    
    return signals

name = "4h_Supertrend_1dVolumeRegime_ATRSize_V1"
timeframe = "4h"
leverage = 1.0