#!/usr/bin/env python3
"""
Experiment #033: 1d Primary + 1w HTF — Simplified Dual Regime with Donchian + RSI

Hypothesis: Previous 1d strategies failed due to TOO MANY filters causing 0 trades.
This version SIMPLIFIES entry conditions while keeping proven elements:
1. Weekly HMA(21) for macro trend bias (proven in #032)
2. Donchian(20) breakout for trend entries (proven in #032)
3. RSI(14) extremes for mean reversion (proven in #024)
4. ATR(14) trailing stop for risk management

Key difference from failed #023/#027: FEWER filters, LOOSER thresholds
- RSI long: <40 (not <15), RSI short: >60 (not >85)
- Donchian: price > 80% of range (not 95%)
- No choppiness index (caused 0 trades in #023)
- No vol spike filter (reduces trade count too much)

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 3.0*ATR trailing (wider for 1d to avoid whipsaw)
Target: 20-40 trades/year on 1d timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_rsi_regime_1w_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA for macro trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate daily indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
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
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(sma_50[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === WEEKLY MACRO BIAS ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Weekly HMA slope (3-bar lookback)
        if i >= 3 and not np.isnan(hma_1w_aligned[i-3]):
            weekly_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-3]
            weekly_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-3]
        else:
            weekly_slope_bull = weekly_bullish
            weekly_slope_bear = weekly_bearish
        
        # === DONCHIAN POSITION ===
        donchian_range = donchian_upper[i] - donchian_lower[i] + 1e-10
        donchian_position = (close[i] - donchian_lower[i]) / donchian_range
        
        # === RSI REGIME ===
        rsi_oversold = rsi_14[i] < 40  # LOOSE threshold for more trades
        rsi_overbought = rsi_14[i] > 60  # LOOSE threshold for more trades
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        
        # === SMA TREND FILTER ===
        price_above_sma50 = close[i] > sma_50[i]
        price_below_sma50 = close[i] < sma_50[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADES) ===
        new_signal = 0.0
        
        # --- LONG ENTRIES ---
        # Trend long: Weekly bullish + Donchian breakout + RSI not overbought
        if weekly_bullish and donchian_position > 0.75 and rsi_14[i] < 65:
            new_signal = POSITION_SIZE
        # Mean reversion long: RSI oversold + price above weekly HMA
        elif rsi_extreme_low and weekly_bullish:
            new_signal = POSITION_SIZE
        # Breakout long: Donchian near high + SMA confirmation
        elif donchian_position > 0.85 and price_above_sma50:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRIES ---
        # Trend short: Weekly bearish + Donchian breakdown + RSI not oversold
        if weekly_bearish and donchian_position < 0.25 and rsi_14[i] > 35:
            new_signal = -POSITION_SIZE
        # Mean reversion short: RSI overbought + price below weekly HMA
        elif rsi_extreme_high and weekly_bearish:
            new_signal = -POSITION_SIZE
        # Breakout short: Donchian near low + SMA confirmation
        elif donchian_position < 0.15 and price_below_sma50:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON WEEKLY TREND REVERSAL ===
        if in_position and position_side > 0:
            if weekly_bearish and weekly_slope_bear:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if weekly_bullish and weekly_slope_bull:
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