#!/usr/bin/env python3
"""
Experiment #462: 12h Primary + 1d/1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 461 failed experiments, clear pattern emerges for 12h:
1. Donchian breakouts work well on SOL (Sharpe +0.782 in research) but need trend filter
2. 1d HMA(21) provides intermediate trend without over-filtering
3. 1w HMA(50) provides major regime bias (bull/bear market detection)
4. RSI(14) filter prevents chasing breakouts at extremes
5. Simpler logic = more trades (critical: need >=30 trades/symbol on train)

Why this might beat current best (Sharpe=0.435):
- Donchian(20) breakout catches sustained moves, not noise
- Dual HTF (1d + 1w) gives better trend confirmation than single HTF
- RSI filter avoids false breakouts at overbought/oversold levels
- 12h TF has lower fee drag than 4h/1h while maintaining edge
- ATR 2.5x trailing stop protects in 2022-style crashes

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_rsi_1d1w_v1"
timeframe = "12h"
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

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Upper = Highest High over period
    Lower = Lowest Low over period
    Middle = (Upper + Lower) / 2
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_keltner(high, low, close, atr_period=10, mult=2.0):
    """
    Calculate Keltner Channel.
    Middle = EMA(20)
    Upper = Middle + mult * ATR
    Lower = Middle - mult * ATR
    """
    close_s = pd.Series(close)
    ema_mid = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema_mid + mult * atr
    lower = ema_mid - mult * atr
    
    return upper, lower, ema_mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators (intermediate trend)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 1w HTF indicators (major regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    keltner_upper, keltner_lower, keltner_mid = calculate_keltner(high, low, close)
    sma_200 = calculate_sma(close, 200)
    
    # HMA on 12h for local trend
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        
        # === 1W MAJOR REGIME (ultra-long-term bias) ===
        # Price above 1w HMA = bull market (favor longs, skip shorts)
        # Price below 1w HMA = bear market (favor shorts, skip longs)
        bull_market = close[i] > hma_1w_21_aligned[i]
        bear_market = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        # 1d HMA(21) > 1d HMA(50) = bullish intermediate trend
        # 1d HMA(21) < 1d HMA(50) = bearish intermediate trend
        trend_1d_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        trend_1d_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 12H LOCAL TREND ===
        trend_12h_bull = hma_12h_21[i] > hma_12h_50[i]
        trend_12h_bear = hma_12h_21[i] < hma_12h_50[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above Donchian upper = potential long
        # Breakout below Donchian lower = potential short
        breakout_long = close[i] > donchian_upper[i-1]  # break above prev high
        breakout_short = close[i] < donchian_lower[i-1]  # break below prev low
        
        # === RSI FILTER (avoid chasing extremes) ===
        # For longs: RSI < 70 (not overbought)
        # For shorts: RSI > 30 (not oversold)
        rsi_ok_long = rsi_14[i] < 70.0
        rsi_ok_short = rsi_14[i] > 30.0
        
        # RSI extreme reversal signals (mean reversion)
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === KELTNER SQUEEZE DETECTION ===
        # Narrow Keltner = low volatility = potential breakout
        keltner_width = (keltner_upper[i] - keltner_lower[i]) / keltner_mid[i]
        is_squeeze = keltner_width < 0.05  # < 5% width = squeeze
        
        # === ENTRY LOGIC — CONFLUENCE BASED ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence conditions)
        if bull_market:  # Only long in bull market regime
            # Condition 1: Donchian breakout + trend confirmation + RSI filter
            if breakout_long and trend_1d_bull and trend_12h_bull and rsi_ok_long:
                new_signal = LONG_SIZE
            # Condition 2: RSI oversold + trend bullish (mean reversion in uptrend)
            elif rsi_oversold and trend_1d_bull and above_sma200:
                new_signal = LONG_SIZE * 0.8
            # Condition 3: Squeeze breakout (volatility expansion)
            elif is_squeeze and breakout_long and trend_12h_bull:
                new_signal = LONG_SIZE
            # Condition 4: HMA crossover + RSI confirmation
            elif trend_12h_bull and rsi_14[i] > 50.0 and rsi_14[i] < 65.0:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (multiple confluence conditions)
        if bear_market:  # Only short in bear market regime
            # Condition 1: Donchian breakdown + trend confirmation + RSI filter
            if breakout_short and trend_1d_bear and trend_12h_bear and rsi_ok_short:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Condition 2: RSI overbought + trend bearish (mean reversion in downtrend)
            elif rsi_overbought and trend_1d_bear and below_sma200:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Condition 3: Squeeze breakdown (volatility expansion)
            elif is_squeeze and breakout_short and trend_12h_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Condition 4: HMA crossover + RSI confirmation
            elif trend_12h_bear and rsi_14[i] < 50.0 and rsi_14[i] > 35.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no position and no signal, use simpler entry conditions
        if not in_position and new_signal == 0.0:
            # Simple long: 12h HMA bullish + RSI moderate
            if bull_market and trend_12h_bull and 40.0 < rsi_14[i] < 60.0:
                new_signal = LONG_SIZE * 0.5
            # Simple short: 12h HMA bearish + RSI moderate
            elif bear_market and trend_12h_bear and 40.0 < rsi_14[i] < 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
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
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and trend_1d_bear:
            new_signal = 0.0
        if in_position and position_side < 0 and trend_1d_bull:
            new_signal = 0.0
        
        # Major regime flip exit (1w HMA cross)
        if in_position and position_side > 0 and bear_market:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_market:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals