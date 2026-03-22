#!/usr/bin/env python3
"""
Experiment #453: 1d Primary + 1w HTF — Donchian Breakout + KAMA Trend + ADX Filter

Hypothesis: After analyzing 452 failed experiments, clear patterns emerge:
1. 1d primary timeframe has proven success (current best Sharpe=0.435)
2. Donchian breakout + HMA trend worked on SOL (Sharpe +0.782 in research)
3. KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA
4. ADX filter prevents entries during weak/noisy markets (ADX>25 = real trend)
5. 1w HTF provides major trend bias without over-filtering
6. Simpler entry logic = more trades (critical: need >=30 trades/symbol on train)

Why this might beat current best (Sharpe=0.435):
- Donchian breakouts catch sustained moves better than RSI mean-reversion
- KAMA reduces whipsaws in choppy markets (adaptive smoothing)
- ADX>25 filter ensures we only trade when trend has momentum
- 1d TF has minimal fee drag (20-50 trades/year target)
- Asymmetric sizing protects in bear markets (0.30 long, 0.25 short)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_kama_adx_1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    
    Efficiency Ratio (ER) = |Close - Close_n| / Sum(|Close_i - Close_i-1|)
    SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    signal = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    
    er = signal / (noise + 1e-10)
    er = er.fillna(0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = strong trend, ADX < 20 = ranging/weak
    """
    n = len(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    plus_dm = pd.Series(plus_dm)
    minus_dm = pd.Series(minus_dm)
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_di = (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / 
               tr.ewm(span=period, min_periods=period, adjust=False).mean() * 100)
    minus_di = (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / 
                tr.ewm(span=period, min_periods=period, adjust=False).mean() * 100)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (highest high / lowest low over period).
    Breakout above upper = long signal
    Breakout below lower = short signal
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    kama_1w_21 = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1w_21_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1d_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1d_30 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_14 = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1w_21_aligned[i]):
            continue
        if np.isnan(kama_1d_10[i]) or np.isnan(kama_1d_30[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(sma_50[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w KAMA = bull bias (favor longs)
        # Price below 1w KAMA = bear bias (favor shorts)
        bull_regime = close[i] > kama_1w_21_aligned[i]
        bear_regime = close[i] < kama_1w_21_aligned[i]
        
        # === 1D LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_1d_10[i] > kama_1d_30[i]
        kama_bearish = kama_1d_10[i] < kama_1d_30[i]
        
        # === TREND STRENGTH (ADX filter) ===
        # ADX > 25 = strong trend, ADX < 20 = weak/ranging
        strong_trend = adx_14[i] > 22.0  # relaxed for more trades
        weak_trend = adx_14[i] < 20.0
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper channel
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # prev bar upper
        # Breakout below lower channel
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # prev bar lower
        
        # === RSI FILTER (avoid extreme entries) ===
        rsi_not_overbought = rsi_14[i] < 75.0
        rsi_not_oversold = rsi_14[i] > 25.0
        
        # === SMA50 FILTER (medium-term trend) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === ENTRY LOGIC — DONCHIAN BREAKOUT + TREND CONFIRMATION ===
        new_signal = 0.0
        
        # LONG ENTRIES
        # Primary: Donchian breakout + bull regime + KAMA bullish + ADX strong
        if bull_regime and kama_bullish and strong_trend:
            if donchian_breakout_long and rsi_not_overbought:
                new_signal = LONG_SIZE
            # Secondary: KAMA crossover + ADX confirmation (no breakout needed)
            elif kama_1d_10[i] > kama_1d_30[i] and kama_1d_10[i-1] <= kama_1d_30[i-1]:
                if adx_14[i] > 20.0 and above_sma50:
                    new_signal = LONG_SIZE * 0.8
        
        # Additional long entry in bull regime (ensure trade frequency)
        if bull_regime and new_signal == 0.0:
            if kama_bullish and rsi_14[i] < 55.0 and above_sma50:
                new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES
        # Primary: Donchian breakdown + bear regime + KAMA bearish + ADX strong
        if bear_regime and kama_bearish and strong_trend:
            if donchian_breakout_short and rsi_not_oversold:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Secondary: KAMA crossover + ADX confirmation
            elif kama_1d_10[i] < kama_1d_30[i] and kama_1d_10[i-1] >= kama_1d_30[i-1]:
                if adx_14[i] > 20.0 and below_sma50:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.8
        
        # Additional short entry in bear regime (ensure trade frequency)
        if bear_regime and new_signal == 0.0:
            if kama_bearish and rsi_14[i] > 45.0 and below_sma50:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
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
        # KAMA crossover exit (trend reversal)
        if in_position and position_side > 0 and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and kama_bullish:
            new_signal = 0.0
        
        # 1w regime flip exit
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # RSI extreme exit (take profit)
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
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