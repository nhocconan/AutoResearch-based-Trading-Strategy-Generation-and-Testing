#!/usr/bin/env python3
"""
Experiment #171: 4h Primary + 1d HTF — Hilbert Transform Cycle Detection + Regime Filter

Hypothesis: Recent failures show that standard indicators (RSI, MACD, ADX) are overfit and
fail in 2025 bear/range market. The Ehlers Hilbert Transform is UNDERUTILIZED in our
experiment history - only 2 strategies tried it vs 20+ with RSI/Fisher.

Hilbert Transform advantages:
- Extracts instantaneous trendline with minimal lag
- Detects cycle phase (where we are in the price cycle)
- Works in BOTH trending and ranging markets (adaptive)
- Mathematically derived from signal processing, not curve-fit

STRATEGY LOGIC:
1) Hilbert Transform gives us: Instantaneous Trendline + Phase
2) Regime Filter: Bollinger Band Width percentile (compression vs expansion)
3) Entry: Phase cross + Trend confirmation + Regime-appropriate logic
   - Compression (BBW < 30th pct): Mean reversion at cycle extremes
   - Expansion (BBW > 70th pct): Trend follow with Hilbert trendline
4) HTF Filter: 1d HMA(21) for macro bias
5) Exit: ATR(14) trailing stop at 2.5x + Phase reversal

POSITION SIZING:
- Full confluence: 0.30
- Partial: 0.20
- Discrete levels to minimize fee churn
- Target: 25-45 trades/year on 4h

Why this might work:
- Hilbert is mathematically robust (not curve-fit like RSI thresholds)
- Regime-adaptive (different logic for chop vs trend)
- Multi-timeframe confirmation reduces false signals
- Conservative sizing protects against 2022-style crashes
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hilbert_cycle_regime_bbw_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hilbert_transform(close, period=18):
    """
    Calculate Ehlers Hilbert Transform Instantaneous Trendline.
    
    Uses quadrature filter to extract instantaneous phase and trendline.
    Based on "Cycle Analytics for Traders" by John Ehlers.
    
    Returns:
    - trendline: Instantaneous trend (less lag than EMA)
    - phase: Cycle phase in degrees (0-360)
    - smooth: Smoothed price for phase calculation
    """
    n = len(close)
    close = np.array(close, dtype=np.float64)
    
    # Detrend using difference from SMA
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    detrend = close - sma
    
    # Handle NaN at start
    detrend[:period] = 0.0
    
    # Hilbert Transform - Quadrature Filter
    # I = InPhase, Q = Quadrature (90-degree shifted)
    I = np.zeros(n)
    Q = np.zeros(n)
    
    for i in range(period, n):
        # I component (current detrended value)
        I[i] = detrend[i]
        
        # Q component (Hilbert transform approximation)
        # Q = 0.0962*detrend[i] + 0.5769*detrend[i-2] - 0.5769*detrend[i-4] - 0.0962*detrend[i-6]
        if i >= 6:
            Q[i] = (0.0962 * detrend[i] + 
                   0.5769 * detrend[i-2] - 
                   0.5769 * detrend[i-4] - 
                   0.0962 * detrend[i-6])
        elif i >= 4:
            Q[i] = 0.0962 * detrend[i] + 0.5769 * detrend[i-2] - 0.5769 * detrend[i-4]
        elif i >= 2:
            Q[i] = 0.0962 * detrend[i] + 0.5769 * detrend[i-2]
        else:
            Q[i] = 0.0962 * detrend[i]
    
    # Smooth I and Q to reduce noise
    I_smooth = pd.Series(I).ewm(span=3, min_periods=3, adjust=False).mean().values
    Q_smooth = pd.Series(Q).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    # Calculate phase angle
    phase = np.zeros(n)
    for i in range(n):
        if np.abs(I_smooth[i]) > 1e-10:
            phase[i] = np.degrees(np.arctan2(Q_smooth[i], I_smooth[i]))
        else:
            phase[i] = phase[i-1] if i > 0 else 0.0
    
    # Ensure phase is continuous (unwrapped)
    for i in range(1, n):
        if phase[i] - phase[i-1] > 180:
            phase[i:] -= 360
        elif phase[i] - phase[i-1] < -180:
            phase[i:] += 360
    
    # Normalize phase to 0-360
    phase = np.mod(phase, 360)
    
    # Calculate Instantaneous Trendline
    # Trendline = I / cos(phase) when phase is not near 90/270
    trendline = np.zeros(n)
    for i in range(period, n):
        cos_phase = np.cos(np.radians(phase[i]))
        if np.abs(cos_phase) > 0.1:
            trendline[i] = sma[i] + I_smooth[i] / cos_phase
        else:
            trendline[i] = sma[i]
    
    return trendline, phase, sma

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    band_width = (upper - lower) / sma
    
    return upper.values, lower.values, band_width.values

def calculate_bbw_percentile(band_width, lookback=100):
    """
    Calculate Bollinger Band Width percentile over lookback period.
    Low percentile = compression (mean reversion regime)
    High percentile = expansion (trend regime)
    """
    n = len(band_width)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = band_width[i-lookback:i]
        current = band_width[i]
        rank = np.sum(window < current)
        percentile[i] = 100.0 * rank / lookback
    
    percentile[:lookback] = 50.0
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    
    # Hilbert Transform
    hilbert_trend, hilbert_phase, hilbert_sma = calculate_hilbert_transform(close, period=18)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_percentile = calculate_bbw_percentile(bb_width, lookback=100)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # Track phase for reversal detection
    prev_phase = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hilbert_trend[i]) or np.isnan(hilbert_phase[i]):
            continue
        if np.isnan(bb_width[i]) or np.isnan(bbw_percentile[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === REGIME DETECTION (Bollinger Band Width Percentile) ===
        bbw_pct = bbw_percentile[i]
        is_compression = bbw_pct < 30.0  # Low BBW = mean reversion regime
        is_expansion = bbw_pct > 70.0    # High BBW = trend regime
        is_transition = not is_compression and not is_expansion
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND (Hilbert) ===
        price_above_hilbert = close[i] > hilbert_trend[i]
        price_below_hilbert = close[i] < hilbert_trend[i]
        
        # === PHASE SIGNALS ===
        current_phase = hilbert_phase[i]
        
        # Phase crossing up through 0/360 = bullish cycle start
        # Phase crossing down through 180 = bearish cycle start
        phase_bullish = 0 <= current_phase <= 45 or current_phase >= 315
        phase_bearish = 135 <= current_phase <= 225
        
        # Phase reversal detection
        phase_crossed_bullish = (prev_phase > 270 and current_phase < 90) if i > 0 else False
        phase_crossed_bearish = (prev_phase < 90 and current_phase > 270) if i > 0 else False
        
        prev_phase = current_phase
        
        # === BOLLINGER BAND POSITION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        
        # === ENTRY LOGIC (REGIME ADAPTIVE) ===
        new_signal = 0.0
        
        if is_compression:
            # MEAN REVERSION REGIME: Fade extremes with phase confirmation
            # Long: Price at BB lower + Phase bullish + 1d bias OK
            if at_bb_lower and phase_bullish and price_above_hma_1d and volume_ok:
                new_signal = POSITION_SIZE_FULL
            # Short: Price at BB upper + Phase bearish + 1d bias OK
            elif at_bb_upper and phase_bearish and price_below_hma_1d and volume_ok:
                new_signal = -POSITION_SIZE_FULL
        
        elif is_expansion:
            # TREND REGIME: Follow Hilbert trendline with phase confirmation
            # Long: Price above Hilbert trend + Phase bullish + 1d bias OK
            if price_above_hilbert and phase_bullish and price_above_hma_1d and volume_ok:
                new_signal = POSITION_SIZE_FULL
            # Short: Price below Hilbert trend + Phase bearish + 1d bias OK
            elif price_below_hilbert and phase_bearish and price_below_hma_1d and volume_ok:
                new_signal = -POSITION_SIZE_FULL
        
        else:
            # TRANSITION regime - reduced size, need stronger confluence
            # Long: Phase crossed bullish + Price above 4h HMA + 1d bias
            if phase_crossed_bullish and price_above_hma_4h and price_above_hma_1d and volume_ok:
                new_signal = POSITION_SIZE_HALF
            # Short: Phase crossed bearish + Price below 4h HMA + 1d bias
            elif phase_crossed_bearish and price_below_hma_4h and price_below_hma_1d and volume_ok:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime/trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above Hilbert trendline
                if price_above_hilbert and price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below Hilbert trendline
                if price_below_hilbert and price_below_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === PHASE REVERSAL EXIT ===
        # Exit long if phase crosses into bearish zone
        if in_position and position_side > 0 and phase_bearish:
            new_signal = 0.0
        
        # Exit short if phase crosses into bullish zone
        if in_position and position_side < 0 and phase_bullish:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below Hilbert trendline
        if in_position and position_side > 0 and price_below_hilbert:
            new_signal = 0.0
        
        # Exit short if price crosses above Hilbert trendline
        if in_position and position_side < 0 and price_above_hilbert:
            new_signal = 0.0
        
        # Exit if macro bias flips against position
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and price_above_hma_1d:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals