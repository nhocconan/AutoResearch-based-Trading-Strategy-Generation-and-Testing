#!/usr/bin/env python3
"""
Experiment #510: 1h Primary + 4h/12h HTF — Regime-Adaptive Mean Reversion with Session Filter

Hypothesis: After 457 failed strategies (mostly CRSI/Choppiness/VolSpike combos), try a 
DIFFERENT approach focusing on REGIME-ADAPTIVE logic with strict session/volume filters
to control trade frequency on 1h timeframe.

Key innovations:
1. CHOPPINESS INDEX regime detection: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
   This is the meta-filter that determines WHICH strategy to use
2. 4h/12h HMA for PRIMARY direction bias (not entry trigger)
3. RSI(7) fast mean reversion with tighter thresholds (25/75 not 30/70)
4. SESSION FILTER: Only trade 8-20 UTC (high liquidity, lower slippage)
5. VOLUME CONFIRMATION: volume > 0.8x 20-bar average (avoid low-liquidity traps)
6. ASYMMETRIC sizing: Long 0.25, Short 0.20 (crypto bias upward long-term)

Why this might beat current best (Sharpe=0.435):
- Regime-adaptive = use right tool for market condition (range vs trend)
- Session filter reduces trades by ~50% (critical for 1h fee drag)
- Volume filter avoids false breakouts on low liquidity
- Different from 448 failed CRSI/Choppiness combos (this uses CHOP as regime switch, not entry)
- 1h TF with strict filters targets 30-60 trades/year (acceptable fee drag 1.5-3%)

Position sizing: 0.20-0.25 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-60 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_session_vol_4h12h_v1"
timeframe = "1h"
leverage = 1.0

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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    highest = high_s.rolling(window=period, min_periods=period).max()
    lowest = low_s.rolling(window=period, min_periods=period).min()
    
    # True range sum
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # CHOP formula
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, 1e-10)
    
    chop = 100.0 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop.values

def calculate_rsi(close, period=7):
    """Calculate RSI with shorter period for faster signals."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def extract_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Extract UTC hour for session filter
    utc_hours = extract_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_7[i]):
            continue
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (utc_hours[i] >= 8) and (utc_hours[i] <= 20)
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_ma_20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        range_regime = chop_14[i] > 55.0  # Mean reversion regime
        trend_regime = chop_14[i] < 45.0  # Trend following regime
        # Neutral regime: 45-55 (no trades or reduced size)
        
        # === 4H/12H TREND BIAS ===
        bull_4h = close[i] > hma_4h_21_aligned[i]
        bear_4h = close[i] < hma_4h_21_aligned[i]
        bull_12h = close[i] > hma_12h_21_aligned[i]
        bear_12h = close[i] < hma_12h_21_aligned[i]
        
        # 4h HMA slope
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # Strong bias when 4h and 12h agree
        strong_bull = bull_4h and bull_12h
        strong_bear = bear_4h and bear_12h
        
        # === RSI SIGNALS (faster 7-period) ===
        rsi_oversold = rsi_7[i] < 25.0
        rsi_overbought = rsi_7[i] > 75.0
        rsi_extreme_low = rsi_7[i] < 15.0
        rsi_extreme_high = rsi_7[i] > 85.0
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = close[i] < bb_lower[i]
        bb_extreme_high = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # Only trade during session with volume confirmation
        if in_session and volume_ok:
            # === RANGE REGIME (CHOP > 55) — MEAN REVERSION ===
            if range_regime:
                # Long: RSI extreme low + below BB + 4h not strongly bearish
                if rsi_extreme_low and bb_extreme_low and not strong_bear:
                    new_signal = LONG_SIZE
                # Long: RSI oversold + 4h bull bias (pullback in uptrend)
                elif rsi_oversold and bull_4h and hma_4h_slope_bull:
                    new_signal = LONG_SIZE * 0.8
                # Short: RSI extreme high + above BB + 4h not strongly bullish
                elif rsi_extreme_high and bb_extreme_high and not strong_bull:
                    new_signal = -SHORT_SIZE
                # Short: RSI overbought + 4h bear bias (bounce in downtrend)
                elif rsi_overbought and bear_4h and hma_4h_slope_bear:
                    new_signal = -SHORT_SIZE * 0.8
            
            # === TREND REGIME (CHOP < 45) — TREND FOLLOWING ===
            elif trend_regime:
                # Long: Strong bull + RSI pullback (not extreme)
                if strong_bull and hma_4h_slope_bull and (25.0 < rsi_7[i] < 50.0):
                    new_signal = LONG_SIZE
                # Long: Strong bull + RSI crossing up from oversold
                elif strong_bull and rsi_7[i] > 30.0 and rsi_7[i-1] <= 30.0:
                    new_signal = LONG_SIZE * 0.8
                # Short: Strong bear + RSI bounce (not extreme)
                elif strong_bear and hma_4h_slope_bear and (50.0 < rsi_7[i] < 75.0):
                    new_signal = -SHORT_SIZE
                # Short: Strong bear + RSI crossing down from overbought
                elif strong_bear and rsi_7[i] < 70.0 and rsi_7[i-1] >= 70.0:
                    new_signal = -SHORT_SIZE * 0.8
            
            # === NEUTRAL REGIME (45 <= CHOP <= 55) — REDUCED ACTIVITY ===
            else:
                # Only take extreme mean reversion signals
                if rsi_extreme_low and bb_extreme_low:
                    new_signal = LONG_SIZE * 0.5
                elif rsi_extreme_high and bb_extreme_high:
                    new_signal = -SHORT_SIZE * 0.5
        
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
        
        # === EXIT CONDITIONS ===
        # Exit long on RSI overbought or regime flip to strong bear
        if in_position and position_side > 0:
            if rsi_overbought or rsi_extreme_high:
                new_signal = 0.0
            if strong_bear and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short on RSI oversold or regime flip to strong bull
        if in_position and position_side < 0:
            if rsi_oversold or rsi_extreme_low:
                new_signal = 0.0
            if strong_bull and hma_4h_slope_bull:
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
                # Flip position
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