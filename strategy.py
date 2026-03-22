#!/usr/bin/env python3
"""
Experiment #565: 1h Primary + 4h/1d HTF — Relaxed Pullback with Volume Confirmation

Hypothesis: After #555 failed (Sharpe=-0.484), the issue was likely:
1. ADX > 20 filter was killing too many valid entries
2. RSI bands (30-50, 50-70) still too narrow for consistent trades across all symbols
3. No volume confirmation = entering on low-volume fakeouts

This strategy uses RELAXED but CONFIRMED entries:
1. 4h HMA(21) for major trend direction (HTF bias) — same as before
2. 1h RSI(14) with WIDER bands: 25-55 long, 45-75 short (ensure trades on all symbols)
3. 1h Volume > 0.6x 20-bar avg (lenient but filters dead zones)
4. 1d HMA(50) as secondary HTF filter (triple-timeframe confluence)
5. ATR(14) 2.5x trailing stop for all positions
6. Position size: 0.25 discrete (appropriate for 1h per Rule 4/10)

Why this might beat Sharpe=0.435:
- REMOVED ADX filter (was killing trades without adding edge)
- WIDER RSI bands = more trades on ALL symbols (BTC, ETH, SOL)
- Added 1d HTF as secondary trend filter (triple-TF confluence)
- Volume filter is LENIENT (0.6x not 0.8x) — confirms interest without overfiltering
- Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test

Position sizing: 0.25 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: Sharpe > 0.5, DD < -30%, trades >= 30/symbol train, >= 3/symbol test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_4h1d_relaxed_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF HMA for major trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF HMA for secondary trend confirmation
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller size for 1h vs 4h (Rule 10: lower TF = more trades = smaller size)
    POSITION_SIZE = 0.25
    POSITION_SIZE_WEAK = 0.15  # Reduced size when 1d trend disagrees
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === 4H MAJOR TREND (primary direction filter) ===
        bull_regime_4h = close[i] > hma_4h_21_aligned[i]
        bear_regime_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope for trend strength
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1D SECONDARY TREND (confirmation filter) ===
        bull_regime_1d = close[i] > hma_1d_50_aligned[i]
        bear_regime_1d = close[i] < hma_1d_50_aligned[i]
        
        # Trend agreement between 4h and 1d
        trend_agree_long = bull_regime_4h and bull_regime_1d
        trend_agree_short = bear_regime_4h and bear_regime_1d
        
        # === VOLUME FILTER (lenient confirmation) ===
        # Volume > 0.6x average = some interest (not dead zone)
        volume_ok = vol_ratio[i] > 0.6
        
        # === RSI PULLBACK ENTRY (WIDER BANDS for more trades) ===
        # Long: RSI 25-55 in uptrend (pullback, not oversold crash)
        # Wider than previous attempts to ensure trades on all symbols
        rsi_pullback_long = 25.0 <= rsi_14[i] <= 55.0
        # Short: RSI 45-75 in downtrend (rally into resistance)
        # Wider than previous attempts to ensure trades on all symbols
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 75.0
        
        # === ENTRY LOGIC — RELAXED but CONFIRMED ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bull + RSI pullback + volume OK
        if bull_regime_4h and rsi_pullback_long and volume_ok:
            # Full size if 1d agrees, reduced if 1d disagrees
            if trend_agree_long:
                new_signal = POSITION_SIZE
            else:
                new_signal = POSITION_SIZE_WEAK
        
        # SHORT ENTRY: 4h bear + RSI pullback + volume OK
        elif bear_regime_4h and rsi_pullback_short and volume_ok:
            # Full size if 1d agrees, reduced if 1d disagrees
            if trend_agree_short:
                new_signal = -POSITION_SIZE
            else:
                new_signal = -POSITION_SIZE_WEAK
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 4h regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_4h:
                new_signal = 0.0
        
        # Exit short on 4h regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_4h:
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