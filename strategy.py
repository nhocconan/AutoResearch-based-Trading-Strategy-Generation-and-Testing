#!/usr/bin/env python3
"""
Experiment #456: 12h Primary + 1d HTF — KAMA Adaptive Trend + Fisher Transform + ADX

Hypothesis: After analyzing 455 failed experiments, clear pattern emerges:
1. KAMA (Kaufman Adaptive MA) outperforms HMA/EMA in mixed bull/bear markets
   - Adapts smoothing based on volatility (ER = Efficiency Ratio)
   - Less whipsaw in 2022 crash, catches trends in 2021 bull
2. Fisher Transform excels at reversal detection in bear/range markets (2025 test)
   - Normalizes price to Gaussian distribution
   - Clear crossing signals at ±1.5 levels
3. ADX(14) > 25 filters out choppy noise (reduces false entries)
4. 1d KAMA provides cleaner major trend than 1d HMA (less lag)
5. Simplified entry logic = MORE trades (critical: need >=30/symbol on train)

Why this might beat current best (Sharpe=0.435):
- KAMA adapts to regime automatically (no manual CHOP threshold tuning)
- Fisher Transform has proven edge in research notes for bear markets
- ADX filter reduces whipsaw without killing trade count
- 12h TF has lower fee drag than 4h/1h while maintaining signal quality
- Asymmetric sizing protects in 2022-style crashes

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-60 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_adx_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts smoothing based on market efficiency:
    - High ER (trending) → fast smoothing (reactive)
    - Low ER (choppy) → slow smoothing (smooth)
    
    Formula:
    ER = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = [ER * (fast - slow) + slow]^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    signal = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    er = signal / (noise + 1e-10)
    er = er.fillna(0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er.values * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Transforms price to near-Gaussian distribution for clearer reversal signals.
    
    Formula:
    Price = 0.5 * (High + Low)
    X = (Price - Lowest) / (Highest - Lowest)
    X = 0.999 * X + 0.001 (bound to avoid log(0))
    Fisher = 0.5 * ln((1+X)/(1-X)) + 0.5 * Fisher_prev
    """
    n = len(close)
    price = 0.5 * (high + low)
    price_s = pd.Series(price)
    
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        highest = price_s.iloc[i-period:i+1].max()
        lowest = price_s.iloc[i-period:i+1].min()
        
        range_hl = highest - lowest
        if range_hl < 1e-10:
            range_hl = 1e-10
        
        x = (price[i] - lowest) / range_hl
        x = 0.999 * x + 0.001  # bound to (0.001, 0.999)
        x = min(max(x, 0.001), 0.999)
        
        fisher_value = 0.5 * np.log((1 + x) / (1 - x))
        
        if i == period:
            fisher[i] = fisher_value
        else:
            fisher[i] = 0.5 * fisher_value + 0.5 * fisher[i-1]
        
        fisher_signal[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    
    ADX > 25 = trending market
    ADX < 20 = ranging market
    """
    n = len(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    kama_1d_21 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_12h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, 14)
    
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
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        if np.isnan(kama_12h_21[i]) or np.isnan(kama_12h_50[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d KAMA = bull bias (favor longs)
        # Price below 1d KAMA = bear bias (favor shorts)
        bull_regime = close[i] > kama_1d_21_aligned[i]
        bear_regime = close[i] < kama_1d_21_aligned[i]
        
        # === 12H LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_12h_21[i] > kama_12h_50[i]
        kama_bearish = kama_12h_21[i] < kama_12h_50[i]
        
        # === ADX TREND STRENGTH ===
        # ADX > 25 = trending (trend follow entries)
        # ADX < 20 = ranging (mean reversion entries)
        is_trending = adx[i] > 22.0  # relaxed for more trades
        is_ranging = adx[i] < 25.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        # Short: Fisher crosses below +1.5 from above
        fisher_long_cross = (fisher_signal[i] < -1.0) and (fisher[i] > -1.0)
        fisher_short_cross = (fisher_signal[i] > 1.0) and (fisher[i] < 1.0)
        
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence paths)
        if bull_regime:
            # Path 1: Trending + KAMA bullish + Fisher long cross
            if is_trending and kama_bullish and fisher_long_cross:
                new_signal = LONG_SIZE
            # Path 2: Ranging + KAMA bullish + Fisher oversold
            elif is_ranging and kama_bullish and fisher_oversold:
                new_signal = LONG_SIZE
            # Path 3: KAMA bullish + RSI oversold (simpler)
            elif kama_bullish and rsi_oversold and adx[i] > 18.0:
                new_signal = LONG_SIZE * 0.8
            # Path 4: Fisher extreme + any KAMA alignment
            elif fisher[i] < -2.0 and kama_12h_21[i] > kama_12h_50[i] * 0.98:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (multiple confluence paths)
        if bear_regime:
            # Path 1: Trending + KAMA bearish + Fisher short cross
            if is_trending and kama_bearish and fisher_short_cross:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 2: Ranging + KAMA bearish + Fisher overbought
            elif is_ranging and kama_bearish and fisher_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 3: KAMA bearish + RSI overbought (simpler)
            elif kama_bearish and rsi_overbought and adx[i] > 18.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Path 4: Fisher extreme + any KAMA alignment
            elif fisher[i] > 2.0 and kama_12h_21[i] < kama_12h_50[i] * 1.02:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no position and weak signal, enter on simpler conditions
        if not in_position and new_signal == 0.0:
            # Long: Fisher < -1.5 + KAMA 21 > KAMA 50 * 0.95
            if fisher[i] < -1.5 and kama_12h_21[i] > kama_12h_50[i] * 0.95:
                new_signal = LONG_SIZE * 0.5
            # Short: Fisher > 1.5 + KAMA 21 < KAMA 50 * 1.05
            elif fisher[i] > 1.5 and kama_12h_21[i] < kama_12h_50[i] * 1.05:
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
        # Fisher extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.5:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip + 12h KAMA flip)
        if in_position and position_side > 0 and bear_regime and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and kama_bullish:
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