#!/usr/bin/env python3
"""
Experiment #377: 1d Primary + 1w HTF — KAMA Adaptive Trend + RSI Pullback

Hypothesis: After 376 experiments, the clearest pattern is:
1. Simple strategies outperform complex dual-regime approaches (most dual-regime failed)
2. 1d primary timeframe generates optimal 20-40 trades/year with minimal fee drag
3. 1w HTF provides superior major trend filter (current best uses 1w)
4. KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA in crypto
5. RSI pullback entries in direction of HTF trend = high probability setups
6. ADX > 25 filter ensures we only trade when trend has momentum

Why this might beat current best (Sharpe=0.435):
- KAMA adapts to chop vs trend automatically (no manual regime switching needed)
- 1w HTF trend filter prevents counter-trend trades (major lesson from 2022 crash)
- RSI pullback (not extreme) = more trades than CRSI extremes, fewer than simple RSI
- ADX confirmation avoids entering during weak/transitioning trends
- ATR trailing stop cuts losers while letting winners run

Position sizing: 0.25-0.30 (discrete, asymmetric long bias)
Stoploss: 2.5 * ATR trailing
Target: 25-40 trades/year on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_pullback_1w_adx_v1"
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
    KAMA adapts smoothing based on market efficiency (trend vs chop).
    ER (Efficiency Ratio) = |price change| / sum of absolute price changes
    Fast SC = 2/(fast_period+1), Slow SC = 2/(slow_period+1)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    price_change = np.abs(close_s.diff(er_period).values)
    sum_changes = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (sum_changes + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed DM
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) for HTF trend."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1d_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Additional filters
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(250, n):  # Start after 200-day SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(kama_1d[i]) or np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter) ===
        # Price above 1w HMA = bull market (favor longs)
        # Price below 1w HMA = bear market (favor shorts)
        weekly_bull = close[i] > hma_1w_21_aligned[i]
        weekly_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D KAMA TREND (adaptive trend detection) ===
        kama_bullish = kama_1d_fast[i] > kama_1d[i]
        kama_bearish = kama_1d_fast[i] < kama_1d[i]
        
        # === ADX TREND STRENGTH ===
        # ADX > 25 = trending market (enter on pullbacks)
        # ADX < 20 = choppy market (stay flat or reduce size)
        trend_strong = adx_14[i] > 25.0
        trend_weak = adx_14[i] < 20.0
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI 35-45 (pullback in uptrend)
        # Short: RSI 55-65 (pullback in downtrend)
        rsi_pullback_long = 35.0 < rsi_14[i] < 50.0
        rsi_pullback_short = 50.0 < rsi_14[i] < 65.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === PRICE POSITION ===
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === VOLATILITY ADJUSTMENT ===
        atr_30 = calculate_atr(high, low, close, 30)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10) if not np.isnan(atr_30[i]) else 1.0
        vol_scale = 0.8 if atr_ratio > 1.5 else 1.0  # Reduce size in high vol
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRIES
        if weekly_bull and trend_strong:
            # Primary: RSI pullback + KAMA bullish
            if rsi_pullback_long and kama_bullish:
                new_signal = LONG_BASE * vol_scale
            # Secondary: RSI oversold + KAMA bullish (stronger signal)
            elif rsi_oversold and kama_bullish:
                new_signal = LONG_STRONG * vol_scale
            # Tertiary: KAMA crossover + ADX rising
            elif kama_bullish and adx_14[i] > adx_14[i-1] and price_above_sma200:
                new_signal = LONG_BASE * 0.8 * vol_scale
        
        elif weekly_bull and not trend_weak:
            # Weaker trend but still bullish weekly
            if rsi_pullback_long and kama_bullish:
                new_signal = LONG_BASE * 0.7 * vol_scale
            elif rsi_oversold:
                new_signal = LONG_BASE * 0.8 * vol_scale
        
        # SHORT ENTRIES (asymmetric - smaller size)
        if weekly_bear and trend_strong:
            # Primary: RSI pullback + KAMA bearish
            if rsi_pullback_short and kama_bearish:
                new_signal = -SHORT_BASE * vol_scale
            # Secondary: RSI overbought + KAMA bearish (stronger signal)
            elif rsi_overbought and kama_bearish:
                new_signal = -SHORT_STRONG * vol_scale
            # Tertiary: KAMA crossover + ADX rising
            elif kama_bearish and adx_14[i] > adx_14[i-1] and not price_above_sma200:
                new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        elif weekly_bear and not trend_weak:
            # Weaker trend but still bearish weekly
            if rsi_pullback_short and kama_bearish:
                new_signal = -SHORT_BASE * 0.7 * vol_scale
            elif rsi_overbought:
                new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 1d) ===
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            # Force entry if no trade for 15 days and conditions are reasonable
            if weekly_bull and rsi_14[i] < 45.0 and kama_bullish:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif weekly_bear and rsi_14[i] > 55.0 and kama_bearish:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_oversold:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and close[i] < kama_1d[i]:
                trend_exit = True
            if position_side < 0 and kama_bullish and close[i] > kama_1d[i]:
                trend_exit = True
        
        # === WEEKLY TREND REVERSAL EXIT ===
        weekly_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and weekly_bear:
                weekly_exit = True
            if position_side < 0 and weekly_bull:
                weekly_exit = True
        
        if stoploss_triggered or rsi_exit or trend_exit or weekly_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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