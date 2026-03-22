#!/usr/bin/env python3
"""
Experiment #532: 4h KAMA Adaptive Crossover with 1d Trend + 1w Regime Filter

Hypothesis: After 500+ failed experiments, key insights for 4h timeframe:
1. KAMA (Kaufman Adaptive MA) outperforms EMA/HMA in ranging markets (2022, 2025)
2. KAMA adjusts speed based on market efficiency ratio - slower in chop, faster in trends
3. 1d HTF provides stable trend bias for 4h entries (not too slow like 1w, not too noisy like 4h)
4. 1w HTF regime filter prevents counter-trend trades in strong bull/bear markets
5. Loose ADX threshold (>20 not >40) ensures sufficient trades while filtering chop
6. Asymmetric sizing: reduce position in bear regime to limit drawdown

Why 4h works:
- Captures 2-5 day trends without 1h noise
- 1d HTF alignment is natural (6x 4h bars per 1d bar)
- Fewer trades = less fee drag, higher quality signals

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d for trend bias, 1w for regime (via mtf_data helper - call ONCE before loop)
Position sizing: 0.25 base, 0.15 in bear regime (asymmetric risk)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_1d_trend_1w_regime_asymmetric_atr_v1"
timeframe = "4h"
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
    Calculate Kaufman Adaptive Moving Average.
    KAMA adjusts smoothing based on market efficiency (trend vs noise).
    ER = 1.0 in strong trend, ER = 0.0 in choppy market.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    # ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    price_change = np.abs(close - np.roll(close, er_period))
    price_change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1))[1:])
    
    volatility[:er_period] = np.nan
    er = price_change / np.where(volatility > 0, volatility, 1e-10)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (trend strength)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_di / atr
    minus_di = 100 * minus_di / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Calculate 1w HTF indicators
    sma_1w_40 = pd.Series(df_1w['close'].values).rolling(window=40, min_periods=40).mean().values
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    sma_1w_40_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_40)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Dual KAMA crossover on 4h
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    kama_slow = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels with asymmetric risk (Rule 4)
    SIZE_BULL = 0.30  # Full size in bull regime
    SIZE_BEAR = 0.15  # Half size in bear regime (reduce risk)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(sma_1w_40_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        
        # === 1W REGIME FILTER ===
        # Bull regime: price above 1w SMA(40)
        # Bear regime: price below 1w SMA(40)
        bull_regime = close[i] > sma_1w_40_aligned[i]
        bear_regime = close[i] < sma_1w_40_aligned[i]
        
        # === 1D TREND BIAS ===
        bull_bias = close[i] > kama_1d_aligned[i]
        bear_bias = close[i] < kama_1d_aligned[i]
        
        # === 4H KAMA CROSSOVER ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === ADX TREND STRENGTH (loose threshold for trades) ===
        adx_strong = adx_14[i] > 20  # Loose threshold to ensure trades
        
        # === RSI FILTER (prevent extreme entries) ===
        rsi_ok_long = rsi_14[i] < 70  # Don't buy when very overbought
        rsi_ok_short = rsi_14[i] > 30  # Don't sell when very oversold
        
        # === ENTRY LOGIC with regime-aware sizing ===
        new_signal = 0.0
        
        # Long: 4h KAMA bullish + 1d bullish bias + ADX strong + RSI ok
        # In bull regime: SIZE_BULL (0.30), in bear regime: SIZE_BEAR (0.15)
        if kama_bullish and bull_bias and adx_strong and rsi_ok_long:
            if bull_regime:
                new_signal = SIZE_BULL
            else:
                new_signal = SIZE_BEAR  # Reduce size in bear regime
        
        # Short: 4h KAMA bearish + 1d bearish bias + ADX strong + RSI ok
        # In bear regime: SIZE_BULL (0.30), in bull regime: SIZE_BEAR (0.15)
        elif kama_bearish and bear_bias and adx_strong and rsi_ok_short:
            if bear_regime:
                new_signal = SIZE_BULL
            else:
                new_signal = SIZE_BEAR  # Reduce size in bull regime
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h KAMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bearish:
                new_signal = 0.0
            if position_side < 0 and kama_bullish:
                new_signal = 0.0
        
        # === 1D BIAS REVERSAL EXIT ===
        # Exit if 1d KAMA bias flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals