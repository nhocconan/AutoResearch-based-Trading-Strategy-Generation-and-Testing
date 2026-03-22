#!/usr/bin/env python3
"""
Experiment #571: 4h Primary + 1d/1w HTF — Donchian Breakout with Regime Filter

Hypothesis: After 50+ failed experiments with complex dual-regime and Connors RSI approaches,
the pattern shows over-filtering kills trades. This strategy uses SIMPLIFIED but PROVEN logic:

1. 1w HMA(21) for MAJOR regime (bull/bear) — only long in bull, only short in bear
2. 1d ADX(14) for trend strength confirmation — avoid choppy periods
3. 4h Donchian(20) breakout for entry timing — catches momentum moves
4. Asymmetric entry: longs only when weekly bullish, shorts only when weekly bearish
5. ATR(14) 2.5x trailing stop for risk management

Why this differs from failed #559-#570:
- NO dual-regime switching (failed in #559, #561, #562)
- NO Connors RSI (failed in #559, #561, #562)
- NO volume filters (failed in #565)
- NO complex Fisher transforms (failed in #568)
- Simple Donchian breakout worked for SOL in past experiments (Sharpe +0.782)
- Weekly regime filter prevents major counter-trend losses in 2022 crash

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 20-50 trades/year on 4h (per Rule 10)
Stoploss: 2.5 * ATR trailing stop (signal → 0 when hit)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_regime_1d1w_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major regime
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF ADX for trend strength
    adx_1d_14 = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_14)
    
    # Calculate 1d HTF HMA for secondary trend filter
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
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
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1W MAJOR REGIME (asymmetric filter) ===
        # Only long when weekly HMA bullish, only short when weekly HMA bearish
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND STRENGTH ===
        # ADX > 20 means some directional movement
        trend_strength_ok = adx_1d_aligned[i] > 20.0
        
        # 1d HMA slope for confirmation
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 4H DONCHIAN BREAKOUT ===
        # Long breakout: price crosses above Donchian upper
        # Short breakout: price crosses below Donchian lower
        prev_close = close[i-1] if i > 0 else close[i]
        
        breakout_long = (prev_close <= donchian_upper[i-1]) and (close[i] > donchian_upper[i])
        breakout_short = (prev_close >= donchian_lower[i-1]) and (close[i] < donchian_lower[i])
        
        # === RSI FILTER (avoid extreme overbought/oversold breakouts) ===
        # Long: RSI < 70 (not extremely overbought)
        # Short: RSI > 30 (not extremely oversold)
        rsi_long_ok = rsi_14[i] < 70.0
        rsi_short_ok = rsi_14[i] > 30.0
        
        # === ENTRY LOGIC — ASYMMETRIC ===
        new_signal = 0.0
        
        # LONG ENTRY: weekly bull + daily bull + ADX ok + Donchian breakout + RSI ok
        if weekly_bull and daily_bull and trend_strength_ok and breakout_long and rsi_long_ok:
            new_signal = POSITION_SIZE
        
        # SHORT ENTRY: weekly bear + daily bear + ADX ok + Donchian breakout + RSI ok
        elif weekly_bear and daily_bear and trend_strength_ok and breakout_short and rsi_short_ok:
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
        # Exit long on weekly regime flip to bear
        if in_position and position_side > 0:
            if weekly_bear:
                new_signal = 0.0
        
        # Exit short on weekly regime flip to bull
        if in_position and position_side < 0:
            if weekly_bull:
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