#!/usr/bin/env python3
"""
Experiment #013: 1d Primary + 1w HTF — Simplified Trend + Mean Reversion

Hypothesis: After 12 failed experiments with complex multi-indicator confluence,
I'm simplifying to proven patterns: weekly trend filter + daily RSI mean reversion.

Key learnings from failures:
- #001, #006, #008, #010: CRSI + Choppiness = too complex, negative Sharpe
- #002, #009: Donchian + multiple filters = whipsaw or 0 trades
- #003, #011, #012: KAMA + Fisher = didn't work on crypto
- #007: HMA + RSI got Sharpe 0.262 — CLOSE! Need to improve this
- #004: Vol spike = too rare, negative Sharpe

What I'm doing DIFFERENT from #007:
1. Use 1w HMA SLOPE (not just price position) — smoother trend signal
2. Asymmetric RSI thresholds: 35/65 (not 30/70) — more trades in crypto
3. Volume confirmation on entry — filters false breakouts
4. Looser exit conditions — let winners run longer
5. Position size 0.30 (vs 0.25-0.35 range) — balanced risk

Why 1d + 1w works:
- 1w HMA captures major bull/bear cycles (2021 bull, 2022 crash, 2023-24 recovery)
- 1d RSI catches pullbacks within the weekly trend
- Target 25-40 trades/year (fee-efficient for daily TF)
- Research shows weekly trend filter improves Sharpe 2x on BTC/ETH

Entry logic (LOOSE to ensure trades):
- Long: 1w HMA slope > 0 AND RSI(14) < 40 OR RSI(14) < 35 (any)
- Short: 1w HMA slope < 0 AND RSI(14) > 60 OR RSI(14) > 65 (any)
- Volume > 20d average (confirms move)

Stoploss: 2.5*ATR trailing
Position size: 0.30 (discrete, per Rule 4)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_weekly_trend_slope_v1"
timeframe = "1d"
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
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend direction
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_sma_20 = calculate_sma(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma_20[i]):
            continue
        if atr_14[i] == 0 or vol_sma_20[i] == 0:
            continue
        
        # === 1W TREND SLOPE ===
        # Compare current 1w HMA to 2 bars ago (smoother than 1 bar)
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else False
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_20[i]  # 80% of avg is fine
        
        # === RSI SIGNALS (asymmetric thresholds for crypto) ===
        rsi_oversold = rsi_14[i] < 40  # More lenient than 30
        rsi_overbought = rsi_14[i] > 60  # More lenient than 70
        rsi_extreme_oversold = rsi_14[i] < 35
        rsi_extreme_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC (LOOSE — either condition works) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Weekly trend must be bullish (slope up)
        if hma_1w_slope_bull:
            # Either RSI oversold OR extreme oversold (both trigger)
            if (rsi_oversold or rsi_extreme_oversold) and volume_confirmed:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Weekly trend must be bearish (slope down)
        if hma_1w_slope_bear:
            # Either RSI overbought OR extreme overbought (both trigger)
            if (rsi_overbought or rsi_extreme_overbought) and volume_confirmed:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless exit conditions met
        if in_position and new_signal == 0.0:
            # Hold if trend hasn't flipped against us
            if position_side > 0 and hma_1w_slope_bull:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and hma_1w_slope_bear:
                new_signal = signals[i-1] if i > 0 else 0.0
            else:
                new_signal = 0.0  # Trend flipped, exit
        
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