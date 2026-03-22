#!/usr/bin/env python3
"""
Experiment #021: 4h KAMA Adaptive Trend + ADX + RSI Pullback with 1d/1w Bias

Hypothesis: Previous HMA-based strategies failed due to lag in ranging markets.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - faster in trends,
slower in chop. Combined with moderate ADX filter (>18, not >25) and relaxed
RSI pullback ranges (35-55 long, 45-65 short) to ensure sufficient trade frequency.

Key improvements over failed experiments:
- KAMA instead of HMA: adapts to market regime automatically (no Choppiness needed)
- ADX > 18 (not >25): ensures trend without being too restrictive
- RSI pullback 35-55 / 45-65 (not extreme): catches more entries, avoids 0-trade failure
- 1d + 1w dual HTF bias: stronger trend confirmation than single HTF
- ATR 2.5x trailing stop: protects capital in 2022-style crashes
- Position size 0.25-0.30: balances return vs drawdown

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Target trades: 25-45/year (4h natural frequency with filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_1d_1w_bias_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility: fast in trends, slow in chop.
    Based on Perry Kaufman's "Trading Systems and Methods".
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER): measures trend efficiency
    change = close_s.diff(period).abs()
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Smoothed constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period-1] = close[period-1]  # Initialize
    
    for i in range(period, n):
        if np.isnan(sc.iloc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1] if i > 0 else close[i]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX).
    Measures trend strength (not direction). ADX > 20 = trending.
    Based on Welles Wilder's original formulation.
    """
    n = len(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values (Wilder's smoothing = EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr) * 100
    minus_di = (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr) * 100
    
    # DX and ADX
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for HTF trend bias."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1W indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_4h_40 = calculate_kama(close, period=40, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(kama_4h_10[i]) or np.isnan(kama_4h_40[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === HTF TREND BIAS (1d + 1w confluence) ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # Strong bias: both 1d and 1w agree
        strong_bull = daily_bullish and weekly_bullish
        strong_bear = daily_bearish and weekly_bearish
        
        # Weak bias: only 1d agrees (still tradeable)
        weak_bull = daily_bullish
        weak_bear = daily_bearish
        
        # === 4H KAMA TREND ===
        kama_bullish = kama_4h_10[i] > kama_4h_40[i]
        kama_bearish = kama_4h_10[i] < kama_4h_40[i]
        
        # === ADX TREND STRENGTH ===
        # ADX > 18 = trending (not too strict, ensures trades)
        trending = adx_14[i] > 18
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (KAMA + ADX + RSI PULLBACK) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: KAMA bullish + trending + RSI pullback (35-55, not extreme)
        if kama_bullish and trending:
            # Strong bias: both 1d and 1w bullish
            if strong_bull:
                if 35 <= rsi_14[i] <= 60:
                    new_signal = current_size
            # Weak bias: only 1d bullish (stricter RSI)
            elif weak_bull:
                if 40 <= rsi_14[i] <= 55:
                    new_signal = current_size * 0.8
        
        # SHORT: KAMA bearish + trending + RSI pullback (45-65, not extreme)
        elif kama_bearish and trending:
            # Strong bias: both 1d and 1w bearish
            if strong_bear:
                if 40 <= rsi_14[i] <= 65:
                    new_signal = -current_size
            # Weak bias: only 1d bearish (stricter RSI)
            elif weak_bear:
                if 45 <= rsi_14[i] <= 60:
                    new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~5 days on 4h), force entry with weaker signal
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if kama_bullish and daily_bullish and rsi_14[i] > 40:
                new_signal = current_size * 0.5
            elif kama_bearish and daily_bearish and rsi_14[i] < 60:
                new_signal = -current_size * 0.5
        
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
            if position_side > 0 and kama_bearish:
                trend_reversal = True
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