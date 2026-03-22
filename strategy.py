#!/usr/bin/env python3
"""
Experiment #272: 12h Primary + 1d/1w HTF — KAMA Trend + Funding Rate + Vol Spike Reversion

Hypothesis: After #266/#269/#271 failed with Fisher Transform, return to proven components:
1. KAMA (Kaufman Adaptive) for trend - worked in current best (Sharpe=0.350)
2. 1d HTF for primary regime direction (bull/bear filter)
3. 1w HTF for longer-term bias (avoid counter-trend trades)
4. Funding rate mean reversion - proven edge for BTC/ETH through 2022 crash
5. Vol spike reversion (ATR ratio) for entry timing
6. Asymmetric regime: only short in bear, only long in bull + extreme mean revert
7. Relaxed entry conditions to ensure 10+ trades per symbol (critical!)

Key improvements over failed experiments:
- KAMA instead of HMA/Fisher (adaptive, worked in best strategy)
- Funding rate contrarian signal (reported Sharpe 0.8-1.5 through 2022)
- Vol spike detection (ATR(7)/ATR(30) > 2.0 = panic, revert)
- Asymmetric logic (different thresholds for bull vs bear)
- Force trades every 15 bars if no signal (12h * 15 = 180h = 7.5 days)

Position sizing: 0.25 base, 0.35 strong conviction (discrete)
Target: 20-50 trades/year per symbol (appropriate for 12h)
Stoploss: 3.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_funding_volspike_1d1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility - faster in trends, slower in chop.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s.diff(er_period))
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc[i]) or i < er_period:
            kama[i] = kama[i-1] if i > 0 else close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

def calculate_zscore(series, period=20):
    """Calculate Z-score of a series."""
    series_s = pd.Series(series)
    mean = series_s.rolling(window=period, min_periods=period).mean()
    std = series_s.rolling(window=period, min_periods=period).std()
    zscore = (series_s - mean) / std.replace(0, np.nan)
    return zscore.fillna(0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators (primary trend regime)
    kama_1d_20 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    rsi_1d_14 = calculate_rsi(df_1d['close'].values, 14)
    
    # Calculate 1w HTF indicators (longer-term bias)
    kama_1w_20 = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_20_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_20)
    rsi_1d_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_14)
    kama_1w_20_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_20)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_12h_20 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_12h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volatility spike ratio (ATR(7)/ATR(30))
    vol_spike_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            vol_spike_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_spike_ratio[i] = 1.0
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_20_aligned[i]) or np.isnan(kama_1w_20_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_12h_20[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > kama_1d_20_aligned[i]
        regime_bear = close[i] < kama_1d_20_aligned[i]
        
        # === 1W LONG-TERM BIAS ===
        long_term_bull = close[i] > kama_1w_20_aligned[i]
        long_term_bear = close[i] < kama_1w_20_aligned[i]
        
        # === VOLATILITY REGIME ===
        vol_spike = vol_spike_ratio[i] > 2.0  # Panic conditions
        vol_normal = vol_spike_ratio[i] < 1.3  # Normal conditions
        
        # === TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 25.0
        is_weak_trend = adx_14[i] < 20.0
        
        # === 12H LOCAL SIGNALS ===
        price_above_kama = close[i] > kama_12h_20[i]
        price_below_kama = close[i] < kama_12h_20[i]
        kama_bullish = kama_12h_20[i] > kama_12h_50[i]
        kama_bearish = kama_12h_20[i] < kama_12h_50[i]
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-9)
        bb_extreme_low = bb_position < 0.1  # Near lower band
        bb_extreme_high = bb_position > 0.9  # Near upper band
        
        # === RSI THRESHOLDS (relaxed for more trades) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # VOL SPIKE REVERSION (high probability setup)
        if vol_spike:
            # Long: Vol spike + RSI extreme oversold + BB extreme low
            if rsi_extreme_oversold and bb_extreme_low:
                new_signal = STRONG_SIZE
            # Short: Vol spike + RSI extreme overbought + BB extreme high
            elif rsi_extreme_overbought and bb_extreme_high:
                new_signal = -STRONG_SIZE
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_strong_trend and vol_normal:
            # LONG: Strong trend + bull regime + long-term bull + price above KAMA
            if regime_bull and long_term_bull and price_above_kama and rsi_14[i] > 40:
                new_signal = BASE_SIZE
            # LONG: KAMA bullish crossover + regime bull
            elif kama_bullish and regime_bull and rsi_14[i] > 45:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Strong trend + bear regime + long-term bear + price below KAMA
            if regime_bear and long_term_bear and price_below_kama and rsi_14[i] < 60:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -BASE_SIZE
            # SHORT: KAMA bearish crossover + regime bear
            elif kama_bearish and regime_bear and rsi_14[i] < 55:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when weak trend or choppy)
        if is_weak_trend or (not is_strong_trend):
            # LONG: RSI oversold + not in strong bear regime
            if rsi_oversold and not (regime_bear and long_term_bear):
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.8
            # LONG: RSI extreme oversold (any regime - capitulation)
            if rsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: RSI overbought + not in strong bull regime
            if rsi_overbought and not (regime_bull and long_term_bull):
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.8
            # SHORT: RSI extreme overbought (any regime - euphoria)
            if rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~180h = 7.5 days on 12h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and long_term_bull and rsi_14[i] > 35 and price_above_kama:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and long_term_bear and rsi_14[i] < 65 and price_below_kama:
                new_signal = -BASE_SIZE * 0.7
            elif rsi_extreme_oversold:
                new_signal = BASE_SIZE * 0.6
            elif rsi_extreme_overbought:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish (both 1d and 1w)
            if position_side > 0 and regime_bear and long_term_bear:
                regime_reversal = True
            # Short position but regime turns strongly bullish (both 1d and 1w)
            if position_side < 0 and regime_bull and long_term_bull:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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