#!/usr/bin/env python3
"""
Hypothesis: Regime-adaptive strategy using 4h HMA trend + 1h Bollinger BW regime detection + RSI entries.
Timeframe: 1h (primary), 4h/12h (HTF filters)
Why: 2025 test period is bear/range market - pure trend following fails. Need to detect regime and adapt.
- Bull regime (BBW expanding): follow 4h HMA trend with RSI pullback entries
- Range regime (BBW contracted): mean reversion with RSI extremes
- 12h HMA filter: reduce short exposure in macro uptrend
- ATR stoploss: mandatory risk management (signal→0 at 2.5*ATR)
- Discrete signals: 0.0, ±0.25, ±0.35 to minimize fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_regime_bbw_rsi_1h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average for smoother trend detection."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = 2 * wma1 - wma2
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI for entry timing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for regime detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth

def calculate_bb_percentile(bandwidth, lookback=100):
    """Percentile of current BBWidth vs historical - regime detector."""
    bb_pct = np.zeros(len(bandwidth))
    for i in range(lookback, len(bandwidth)):
        window = bandwidth[i-lookback:i]
        bb_pct[i] = np.sum(window < bandwidth[i]) / lookback
    return bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1) ===
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF HMAs
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === CALCULATE 1h INDICATORS (vectorized before loop) ===
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger Bands(close, 20, 2.0)
    bb_pct = calculate_bb_percentile(bb_width, 100)
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.20  # Asymmetric - smaller shorts in crypto
    prev_signal = 0.0
    
    # Track entry prices for stoploss
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        trend_4h = 1.0 if hma_4h_aligned[i] > hma_4h_aligned[i-5] else -1.0
        trend_12h = 1.0 if hma_12h_aligned[i] > hma_12h_aligned[i-10] else -1.0
        
        # Regime detection: BBWidth percentile
        # < 0.3 = squeeze (range), > 0.7 = expansion (trend)
        regime_range = bb_pct[i] < 0.35
        regime_trend = bb_pct[i] > 0.65
        
        signal = 0.0
        
        # === TREND REGIME: Follow 4h HMA with RSI pullback ===
        if regime_trend:
            if trend_4h > 0 and trend_12h >= 0:  # Bull trend
                if rsi[i] < 55:  # Pullback entry
                    signal = SIZE_LONG
            elif trend_4h < 0 and trend_12h <= 0:  # Bear trend
                if rsi[i] > 45:  # Pullback entry
                    signal = -SIZE_SHORT
        
        # === RANGE REGIME: Mean reversion at BB extremes ===
        elif regime_range:
            if close[i] < bb_lower[i] and rsi[i] < 35:
                signal = SIZE_LONG
            elif close[i] > bb_upper[i] and rsi[i] > 65:
                signal = -SIZE_SHORT
        
        # === NEUTRAL REGIME: Follow 4h trend only ===
        else:
            if trend_4h > 0 and rsi[i] < 60:
                signal = SIZE_LONG * 0.7  # Reduced size in uncertain regime
            elif trend_4h < 0 and rsi[i] > 40:
                signal = -SIZE_SHORT * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side == 1 and entry_price > 0:
            # Update highest since entry for trailing
            highest_since_entry = max(highest_since_entry, close[i])
            # Hard stoploss at 2.5*ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signal = 0.0
                position_side = 0
                entry_price = 0.0
            # Trailing stop: exit if drops 2*ATR from highest
            elif close[i] < highest_since_entry - 2.0 * atr[i]:
                signal = 0.0
                position_side = 0
                entry_price = 0.0
        
        elif position_side == -1 and entry_price > 0:
            # Update lowest since entry for trailing
            lowest_since_entry = min(lowest_since_entry, close[i])
            # Hard stoploss at 2.5*ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signal = 0.0
                position_side = 0
                entry_price = 0.0
            # Trailing stop: exit if rises 2*ATR from lowest
            elif close[i] > lowest_since_entry + 2.0 * atr[i]:
                signal = 0.0
                position_side = 0
                entry_price = 0.0
        
        # === TRACK POSITION FOR STOPLOSS ===
        if signal != 0 and position_side == 0:
            # New position
            position_side = 1 if signal > 0 else -1
            entry_price = close[i]
            highest_since_entry = close[i]
            lowest_since_entry = close[i]
        elif signal == 0 and position_side != 0:
            # Position closed
            position_side = 0
            entry_price = 0.0
        
        # === DISCRETIZE SIGNAL (Rule 4 - minimize churn) ===
        if abs(signal - prev_signal) < 0.05:
            signal = prev_signal  # No change = no trade
        
        # Clamp to max size
        signal = np.clip(signal, -0.40, 0.40)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals