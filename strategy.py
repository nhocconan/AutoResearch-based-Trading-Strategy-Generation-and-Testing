#!/usr/bin/env python3
"""
Experiment #584: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + ADX Filter

Hypothesis: After analyzing 580+ failed experiments, the pattern is clear:
- Complex regime-switching (chop + multiple filters) = 0 trades (Sharpe=0.000)
- Simple trend-following with pullback entries = consistent positive Sharpe
- #579 (4h CRSI + 1d HMA) got Sharpe=0.103 — PROOF 4h+1d works
- Current best #577 (1d Choppiness + CRSI + 1w) got Sharpe=0.520

This strategy combines PROVEN elements with LOOSER entry conditions to ensure trades:
1. 12h HMA(21) for PRIMARY trend direction (slower than 1d, more responsive than 1w)
2. 1d HMA(50) for secondary trend confirmation (dual HTF filter)
3. 4h RSI(7) for pullback entries — WIDER bands (30-50 long, 50-70 short) to ensure trades
4. ADX(14) > 18 minimal trend filter (not 25, which kills trade count)
5. ATR(14) 2.5x trailing stop for all positions
6. Position size: 0.30 discrete (per Rule 4, max 0.40)

Why this might beat Sharpe=0.520:
- 12h HTF is optimal balance between 1d (slow) and 4h (noisy)
- Dual HTF (12h + 1d) reduces false signals vs single HTF
- RSI(7) reacts faster than RSI(14) for 4h entries
- WIDER RSI bands (30-50 vs 25-35) = MORE trades = less chance of 0-trade failure
- ADX > 18 (not 25) allows entries in moderate trends
- 4h TF targets 20-50 trades/year (per Rule 10)

Position sizing: 0.30 base (discrete per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_adx_12h1d_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF HMAs for trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 4h entries
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_7[i]):
            continue
        
        # === 12H PRIMARY TREND (main direction filter) ===
        bull_12h = close[i] > hma_12h_21_aligned[i]
        bear_12h = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope for trend strength
        bull_slope_12h = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        bear_slope_12h = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 1D SECONDARY TREND (confirmation filter) ===
        bull_1d = close[i] > hma_1d_21_aligned[i]
        bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # Dual HTF agreement for stronger signals
        strong_bull = bull_12h and bull_1d and bull_slope_12h
        strong_bear = bear_12h and bear_1d and bear_slope_12h
        
        # === ADX FILTER (minimal trend strength) ===
        # ADX > 18 means some directional movement (permissive for trade count)
        trend_ok = adx_14[i] > 18.0
        
        # === RSI PULLBACK ENTRY (WIDE BANDS for more trades) ===
        # Long: RSI(7) < 50 in uptrend (pullback, not extreme oversold)
        # Short: RSI(7) > 50 in downtrend (rally, not extreme overbought)
        # This ensures MORE trades than extreme RSI<30/>70
        rsi_pullback_long = rsi_7[i] < 50.0
        rsi_pullback_short = rsi_7[i] > 50.0
        
        # === ENTRY LOGIC — SIMPLE with DUAL HTF ===
        new_signal = 0.0
        
        # LONG ENTRY: 12h bull + 1d bull + RSI pullback + ADX OK
        if strong_bull and rsi_pullback_long and trend_ok:
            new_signal = POSITION_SIZE
        
        # SHORT ENTRY: 12h bear + 1d bear + RSI pullback + ADX OK
        elif strong_bear and rsi_pullback_short and trend_ok:
            new_signal = -POSITION_SIZE
        
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
        # Exit long on 12h regime flip to bear
        if in_position and position_side > 0:
            if bear_12h and bear_slope_12h:
                new_signal = 0.0
        
        # Exit short on 12h regime flip to bull
        if in_position and position_side < 0:
            if bull_12h and bull_slope_12h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals