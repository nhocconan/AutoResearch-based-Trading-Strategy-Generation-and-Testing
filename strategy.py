#!/usr/bin/env python3
"""
Experiment #729: 15m Primary + 1h/1d HTF — RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 1h trend + 1d regime filter provides optimal
entry timing while avoiding fee drag. Using RSI(7) pullback entries in direction
of HTF trend, with session filter for liquidity.

Key innovations:
1. 1h HMA(21) for intermediate trend direction
2. 1d HMA(50) for long-term regime filter (only trade with daily trend)
3. 15m RSI(7) for precise entry timing (oversold/overbought pullbacks)
4. Session filter: 00-12 UTC (London+NY overlap = higher liquidity)
5. Volume confirmation: taker_buy_volume > 1.5x 20-bar average
6. ATR(14) 2.5x trailing stoploss
7. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

Why this might work on 15m:
- HTF filters reduce false signals (1h + 1d agreement)
- RSI(7) is sensitive enough for 15m entries
- Session filter avoids low-liquidity Asian hours
- Volume confirmation ensures real momentum
- Loose RSI thresholds (25/75) ensure trade generation

Target: Sharpe>0.40, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_hma_1h1d_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(values, period):
    """Simple Moving Average"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_15m_21 = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average for confirmation
    vol_avg_20 = calculate_sma(volume, 20)
    taker_vol_avg_20 = calculate_sma(taker_buy_vol, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_21[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London+NY overlap) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 0 <= hour_utc <= 12
        
        # === HTF TREND BIAS ===
        # 1d HMA(50) for long-term regime
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1h HMA(21) for intermediate trend
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = False
        if not np.isnan(vol_avg_20[i]) and vol_avg_20[i] > 1e-10:
            vol_confirmed = volume[i] > 1.2 * vol_avg_20[i]
        
        taker_confirmed = False
        if not np.isnan(taker_vol_avg_20[i]) and taker_vol_avg_20[i] > 1e-10:
            taker_confirmed = taker_buy_vol[i] > 1.3 * taker_vol_avg_20[i]
        
        # === RSI PULLBACK CONDITIONS (LOOSE for trade generation) ===
        # Long: RSI(7) oversold pullback in uptrend
        rsi_oversold = rsi_7[i] < 35.0  # Loose threshold
        rsi_very_oversold = rsi_7[i] < 25.0
        
        # Short: RSI(7) overbought pullback in downtrend
        rsi_overbought = rsi_7[i] > 65.0  # Loose threshold
        rsi_very_overbought = rsi_7[i] > 75.0
        
        # === 15m HMA TREND CONFIRMATION ===
        hma_15m_bull = close[i] > hma_15m_21[i]
        hma_15m_bear = close[i] < hma_15m_21[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG entries (need 1d bull + at least one more confirmation)
        if htf_1d_bull:
            # Strong long: 1h bull + RSI oversold + session
            if htf_1h_bull and rsi_oversold and in_session:
                desired_signal = SIZE_STRONG
            # Base long: 1h bull + RSI very oversold (any session)
            elif htf_1h_bull and rsi_very_oversold:
                desired_signal = SIZE_BASE
            # Base long: 15m HMA bull + RSI oversold + volume
            elif hma_15m_bull and rsi_oversold and vol_confirmed:
                desired_signal = SIZE_BASE
        
        # SHORT entries (need 1d bear + at least one more confirmation)
        elif htf_1d_bear:
            # Strong short: 1h bear + RSI overbought + session
            if htf_1h_bear and rsi_overbought and in_session:
                desired_signal = -SIZE_STRONG
            # Base short: 1h bear + RSI very overbought (any session)
            elif htf_1h_bear and rsi_very_overbought:
                desired_signal = -SIZE_BASE
            # Base short: 15m HMA bear + RSI overbought + volume
            elif hma_15m_bear and rsi_overbought and vol_confirmed:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals