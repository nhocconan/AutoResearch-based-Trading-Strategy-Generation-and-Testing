#!/usr/bin/env python3
"""
Experiment #153: 1d Primary + 1w HTF — KAMA Adaptive Trend + RSI Mean Reversion

Hypothesis: Previous 1d strategies failed due to overly complex regime filters
and strict entry conditions (0 trades). This strategy simplifies to:

1. KAMA(14) adaptive trend — adjusts to volatility, works in both trend/range
2. 1w HMA(21) slope — major trend bias (only trade with weekly trend)
3. RSI(14) extremes — entry trigger at 30/70 levels (not too strict)
4. ATR(14) stoploss — 2.5x trailing stop
5. Time-based exit — force exit after 60 bars if no stop hit

Why this should work:
- KAMA adapts to market conditions (ER = Efficiency Ratio)
- 1w HTF provides stable trend filter (less noise than 1d)
- RSI 30/70 thresholds generate sufficient trades (not 20/80 which is too strict)
- 1d timeframe = 20-50 trades/year target
- Simple logic = fewer failure points

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise using Efficiency Ratio.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close_s.diff().values)
    volatility = pd.Series(close).diff(period).abs().values
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    
    er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / np.abs(hma_values[i - lookback]) * 100
    return slope

def calculate_donchian_channels(high, low, period=20):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    kama_14 = calculate_kama(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    entry_bar = 0
    bars_in_trade = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === 1W TREND BIAS (primary filter) ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        weekly_neutral = not weekly_bullish and not weekly_bearish
        
        price_above_kama = close[i] > kama_14[i]
        price_below_kama = close[i] < kama_14[i]
        
        # === RSI CONDITIONS (entry triggers) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i] * 0.995
        donchian_breakout_down = close[i] < donchian_lower[i] * 1.005
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if weekly_neutral:
            current_size = HALF_SIZE
        
        # === ENTRY LOGIC (simplified for more trades) ===
        new_signal = 0.0
        
        # LONG ENTRIES
        long_condition_1 = weekly_bullish and rsi_oversold and price_below_kama
        long_condition_2 = weekly_bullish and rsi_extreme_low
        long_condition_3 = weekly_neutral and rsi_extreme_low and price_below_kama
        long_condition_4 = donchian_breakout_up and weekly_bullish and rsi_14[i] < 55
        
        if long_condition_1 or long_condition_2 or long_condition_3 or long_condition_4:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_condition_1 = weekly_bearish and rsi_overbought and price_above_kama
        short_condition_2 = weekly_bearish and rsi_extreme_high
        short_condition_3 = weekly_neutral and rsi_extreme_high and price_above_kama
        short_condition_4 = donchian_breakout_down and weekly_bearish and rsi_14[i] > 45
        
        if short_condition_1 or short_condition_2 or short_condition_3 or short_condition_4:
            new_signal = -current_size
        
        # === TIME-BASED EXIT (force exit after 60 bars) ===
        bars_in_trade = i - entry_bar if in_position else 0
        time_exit = bars_in_trade > 60
        
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
            if position_side > 0 and weekly_bearish:
                trend_reversal = True
            if position_side < 0 and weekly_bullish:
                trend_reversal = True
        
        if stoploss_triggered or time_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                entry_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                entry_bar = 0
        
        signals[i] = new_signal
    
    return signals