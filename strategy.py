#!/usr/bin/env python3
"""
Experiment #031: 4h KAMA Adaptive Trend with 1d Bias and Donchian Breakout

Hypothesis: Previous 4h strategies failed due to either too many filters (killing trades)
or simple EMA crossovers (whipsaw in 2022 crash). KAMA (Kaufman Adaptive Moving Average)
adapts smoothing based on market efficiency - smooth in noise, responsive in trends.

Key components:
1. 4h KAMA(10) for adaptive trend following (responds faster in trending markets)
2. 1d HMA(21) for major trend bias (only trade in direction of daily trend)
3. Donchian(20) breakout for entry timing (breakout + trend = higher probability)
4. RSI(14) filter to avoid extreme overbought/oversold entries
5. ATR(14) trailing stoploss at 2.5x for risk management

Why this should work better:
- KAMA adapts to volatility (unlike fixed EMA/HMA)
- 1d bias prevents counter-trend trades that failed in 2022
- Donchian breakout ensures we enter on momentum, not pullbacks (pullbacks = 0 trades)
- Simple 3-condition entry (trend + breakout + bias) = sufficient trades
- 4h timeframe = natural 20-50 trades/year (fee drag manageable)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_donchian_1d_bias_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    ER = 1 in strong trend, ER = 0 in choppy market.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio: |change| / sum(|changes|)
        if i >= efficiency_period:
            signal = np.abs(close[i] - close[i - efficiency_period])
            noise = np.sum(np.abs(np.diff(close[i - efficiency_period:i + 1])))
            if noise > 0:
                er = signal / noise
            else:
                er = 0.0
        else:
            er = 0.0
        
        # Smoothing constant: ER * (fast_sc - slow_sc) + slow_sc
        fast_sc = 2.0 / (fast_period + 1.0)
        slow_sc = 2.0 / (slow_period + 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA = prior_KAMA + sc * (price - prior_KAMA)
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h_10 = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(kama_4h_10[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H KAMA TREND ===
        # KAMA above/below price indicates trend direction
        kama_bullish = close[i] > kama_4h_10[i]
        kama_bearish = close[i] < kama_4h_10[i]
        
        # KAMA slope confirmation
        kama_slope_up = kama_4h_10[i] > kama_4h_10[i - 5] if i >= 5 else False
        kama_slope_down = kama_4h_10[i] < kama_4h_10[i - 5] if i >= 5 else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 and not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 and not np.isnan(donchian_lower[i - 1]) else False
        
        # === RSI FILTER (avoid extreme entries) ===
        rsi_ok_long = rsi_14[i] < 75  # Don't long at extreme overbought
        rsi_ok_short = rsi_14[i] > 25  # Don't short at extreme oversold
        
        # === POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i - 100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === LONG ENTRY: KAMA bullish + breakout + daily bias + RSI ok ===
        # Need 3 of 4 conditions for entry (flexible but not too loose)
        long_conditions = 0
        if kama_bullish:
            long_conditions += 1
        if breakout_long:
            long_conditions += 1
        if daily_bullish:
            long_conditions += 1
        if rsi_ok_long:
            long_conditions += 1
        
        # Enter long if 3+ conditions met AND breakout occurred
        if long_conditions >= 3 and breakout_long and kama_bullish:
            new_signal = current_size
        
        # === SHORT ENTRY: KAMA bearish + breakout + daily bias + RSI ok ===
        short_conditions = 0
        if kama_bearish:
            short_conditions += 1
        if breakout_short:
            short_conditions += 1
        if daily_bearish:
            short_conditions += 1
        if rsi_ok_short:
            short_conditions += 1
        
        # Enter short if 3+ conditions met AND breakout occurred
        if short_conditions >= 3 and breakout_short and kama_bearish:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow 2-condition entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if kama_bullish and daily_bullish and rsi_14[i] < 65:
                new_signal = current_size * 0.7
            elif kama_bearish and daily_bearish and rsi_14[i] > 35:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if KAMA turns bearish (price crosses below KAMA)
            if position_side > 0 and kama_bearish:
                trend_reversal = True
            # Exit short if KAMA turns bullish (price crosses above KAMA)
            if position_side < 0 and kama_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals