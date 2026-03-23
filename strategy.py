#!/usr/bin/env python3
"""
Experiment #083: 1d Primary + 1w HTF — Simple Trend-Follow with RSI Pullbacks

Hypothesis: Simpler is better for daily timeframe. Use 1w HMA(21) for macro trend bias,
1d RSI(7) for pullback entries. Avoid complex regime-switching that caused 0 trades in #073-#075.

Key design:
1) 1w HMA(21) slope determines trend direction — only trade with weekly trend
2) 1d RSI(7) pullback entries — RSI<45 for long, RSI>55 for short (loose enough for trades)
3) ATR(14) trailing stoploss at 2.5x — protects capital in crashes
4) No funding filter (failed in #079, #082) — keeps logic simple
5) Discrete sizing: 0.30 base position

Why this should work:
- 1d timeframe proven to work (research shows 20-50 trades/year optimal)
- Weekly trend filter prevents counter-trend trades in 2022 crash and 2025 bear
- RSI(7) more responsive than RSI(14) for daily entries
- Simple logic = fewer bugs, more robust across BTC/ETH/SOL
- Loose entry conditions ensure >=30 trades on train

Position size: 0.30 (discrete)
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_pullback_1w_hma_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1w HMA slope (trend strength)
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] != 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_7[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            continue
        if atr_14[i] == 0 or np.isnan(atr_14[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Weekly trend strength (slope threshold)
        weekly_uptrend = hma_1w_slope[i] > 0.5  # >0.5% weekly gain
        weekly_downtrend = hma_1w_slope[i] < -0.5  # <-0.5% weekly loss
        
        # === RSI PULLBACK SIGNALS (loose thresholds for trade frequency) ===
        # Long: RSI(7) pullback in uptrend
        rsi_oversold = rsi_7[i] < 45.0
        rsi_extreme_oversold = rsi_7[i] < 35.0
        
        # Short: RSI(7) pullback in downtrend
        rsi_overbought = rsi_7[i] > 55.0
        rsi_extreme_overbought = rsi_7[i] > 65.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Weekly uptrend + RSI pullback ---
        if price_above_hma_1w and weekly_uptrend:
            # Standard long: RSI pullback + EMA bullish
            if rsi_oversold and ema_bullish:
                new_signal = POSITION_SIZE
            # Strong long: Extreme RSI oversold (catch panic dips)
            elif rsi_extreme_oversold:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Weekly downtrend + RSI pullback ---
        if price_below_hma_1w and weekly_downtrend:
            # Standard short: RSI pullback + EMA bearish
            if rsi_overbought and ema_bearish:
                new_signal = -POSITION_SIZE
            # Strong short: Extreme RSI overbought (catch rally tops)
            elif rsi_extreme_overbought:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if RSI hasn't reached exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 30.0:
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
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if weekly HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_1w and weekly_downtrend:
                new_signal = 0.0
        
        # Exit short if weekly HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_1w and weekly_uptrend:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
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