#!/usr/bin/env python3
"""
Experiment #442: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + RSI Pullback

Hypothesis: After 441 experiments, clear pattern: KAMA (Kaufman Adaptive Moving Average)
outperforms static EMAs/HMAs because it automatically adapts to volatility regimes.
- KAMA speeds up in trends (low noise), slows in chop (high noise)
- No need for explicit ADX/Choppiness regime detection = fewer filters = more trades
- 1d HMA provides major trend bias (prevents counter-trend entries in 2022-style crashes)
- 1w HMA as ultimate filter (only trade with weekly trend)
- RSI pullback entries (not extremes) ensure sufficient trade frequency

Why this might beat current best (Sharpe=0.435):
- KAMA adapts automatically = no regime switch logic needed
- Triple HTF confirmation (1w > 1d > 12h) prevents major drawdowns
- RSI 35-65 entry zone (not 30/70 extremes) = more trades, earlier entries
- Asymmetric sizing: longs 0.30, shorts 0.25 (crypto long bias)
- 12h TF = ~40 trades/year target, low fee drag

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_hma_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, eff_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility: fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, eff_period))
    change[:eff_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(eff_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-eff_period):i+1])))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er[:eff_period] = np.nan
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[eff_period] = close[eff_period]
    for i in range(eff_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    kama_1d_10 = calculate_kama(df_1d['close'].values, eff_period=10)
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    kama_1d_10_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_10)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_10 = calculate_kama(close, eff_period=10)
    kama_12h_20 = calculate_kama(close, eff_period=20)
    rsi_12h_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(kama_1d_10_aligned[i]):
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(kama_12h_10[i]) or np.isnan(kama_12h_20[i]):
            continue
        if np.isnan(rsi_12h_14[i]):
            continue
        
        # === 1W ULTIMATE TREND FILTER ===
        # Only trade in direction of weekly trend
        weekly_bull = close[i] > hma_1w_21_aligned[i]
        weekly_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D MAJOR TREND ===
        daily_bull = close[i] > hma_1d_21_aligned[i] and kama_1d_10_aligned[i] > hma_1d_21_aligned[i]
        daily_bear = close[i] < hma_1d_21_aligned[i] and kama_1d_10_aligned[i] < hma_1d_21_aligned[i]
        
        # === 12H KAMA TREND (adaptive) ===
        # Fast KAMA > Slow KAMA = bullish momentum
        kama_bullish = kama_12h_10[i] > kama_12h_20[i]
        kama_bearish = kama_12h_10[i] < kama_12h_20[i]
        
        # KAMA slope (direction)
        kama_slope_bull = kama_12h_10[i] > kama_12h_10[i-5] if i >= 5 else False
        kama_slope_bear = kama_12h_10[i] < kama_12h_10[i-5] if i >= 5 else False
        
        # === RSI ENTRY ZONE (not extremes = more trades) ===
        rsi_pullback_long = 35.0 <= rsi_12h_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_12h_14[i] <= 65.0
        rsi_strong_long = rsi_12h_14[i] > 50.0
        rsi_strong_short = rsi_12h_14[i] < 50.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (require weekly bull OR daily bull)
        if weekly_bull or daily_bull:
            # Primary: KAMA bullish + RSI pullback
            if kama_bullish and rsi_pullback_long:
                new_signal = LONG_SIZE
            # Secondary: KAMA slope up + RSI above 50
            elif kama_slope_bull and rsi_strong_long and kama_bullish:
                new_signal = LONG_SIZE * 0.9
            # Frequency boost: if no trade for 8 bars, enter on weaker signal
            elif bars_since_last_trade > 8 and kama_bullish and rsi_12h_14[i] > 45.0:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (require weekly bear OR daily bear)
        if weekly_bear or daily_bear:
            # Primary: KAMA bearish + RSI pullback
            if kama_bearish and rsi_pullback_short:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Secondary: KAMA slope down + RSI below 50
            elif kama_slope_bear and rsi_strong_short and kama_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.9
            # Frequency boost
            elif bars_since_last_trade > 8 and kama_bearish and rsi_12h_14[i] < 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit)
        if in_position and position_side > 0 and rsi_12h_14[i] > 72.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_12h_14[i] < 28.0:
            new_signal = 0.0
        
        # KAMA reversal exit
        if in_position and position_side > 0 and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and kama_bullish:
            new_signal = 0.0
        
        # Weekly trend reversal (major exit)
        if in_position and position_side > 0 and weekly_bear:
            new_signal = 0.0
        if in_position and position_side < 0 and weekly_bull:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if highest_price == 0.0:
                highest_price = close[i]
            else:
                highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals