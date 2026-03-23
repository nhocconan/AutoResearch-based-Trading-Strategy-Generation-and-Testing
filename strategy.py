#!/usr/bin/env python3
"""
Experiment #758: 30m Primary + 4h/1d HTF — RSI Pullback + Volume + Session Filter

Hypothesis: After analyzing failures #748, #750 (0 trades) and success of #751 (Sharpe=0.342):
1. 30m timeframe needs STRICT filters to avoid >100 trades/year (fee drag)
2. CRSI caused 0 trades in some experiments - use simpler RSI(14) with looser thresholds
3. Volume confirmation (taker_buy_ratio) adds edge for entry timing
4. Session filter (8-20 UTC) captures high-liquidity periods, reduces noise
5. 4h HMA(21) for trend direction, 1d HMA(21) for regime bias
6. ATR(14) trailing stop 2.5x for risk management
7. Target: 40-80 trades/year, Sharpe > 0.612, ALL symbols positive

Strategy design:
1. 1d HMA(21) aligned - primary regime bias (bull/bear)
2. 4h HMA(21) aligned - intermediate trend direction
3. 30m RSI(14) - entry timing (oversold <35 long, overbought >65 short)
4. 30m Volume ratio (taker_buy/volume) - confirmation >0.52 long, <0.48 short
5. Session filter: only 8-20 UTC (high liquidity hours)
6. ATR(14) trailing stop 2.5x
7. Discrete signals: 0.0, ±0.20, ±0.25

Key improvements from #751:
- Lower TF (30m) for more entry opportunities within HTF trend
- Volume confirmation reduces false entries
- Session filter eliminates low-liquidity noise (Asian overnight)
- Simpler RSI(14) instead of CRSI (more reliable, proven edge)
- Stricter confluence (3+ filters) to control trade frequency

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_vol_session_hma_4h1d_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - standard implementation."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio - measures buying pressure."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = taker_buy_volume / (volume + 1e-10)
    return ratio

def calculate_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    # Convert to hours UTC
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    sma_50_30m = calculate_sma(close, period=50)
    sma_200_30m = calculate_sma(close, period=200)
    vol_ratio_30m = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract hour for session filter
    hours = np.array([calculate_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50_30m[i]) or np.isnan(sma_200_30m[i]):
            continue
        if np.isnan(vol_ratio_30m[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HTF HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === RSI SIGNALS (30m) ===
        rsi_oversold = rsi_30m[i] < 35
        rsi_overbought = rsi_30m[i] > 65
        rsi_extreme_low = rsi_30m[i] < 25
        rsi_extreme_high = rsi_30m[i] > 75
        
        # === VOLUME CONFIRMATION ===
        vol_buying = vol_ratio_30m[i] > 0.52
        vol_selling = vol_ratio_30m[i] < 0.48
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50_30m[i]
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma50 = close[i] < sma_50_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (3+ confluence required) ===
        # Must be in session, need HTF trend + RSI + volume confirmation
        if in_session:
            # Strong long: 1d bullish + 4h bullish + RSI oversold + volume buying
            if trend_1d_bullish and trend_4h_bullish and rsi_oversold and vol_buying:
                desired_signal = BASE_SIZE
            
            # Moderate long: 1d bullish + RSI extreme + above SMA50
            elif trend_1d_bullish and rsi_extreme_low and above_sma50:
                desired_signal = REDUCED_SIZE
            
            # Pullback long: 4h bullish + RSI oversold + above SMA200
            elif trend_4h_bullish and rsi_oversold and above_sma200 and vol_buying:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS (3+ confluence required) ===
        if in_session:
            # Strong short: 1d bearish + 4h bearish + RSI overbought + volume selling
            if trend_1d_bearish and trend_4h_bearish and rsi_overbought and vol_selling:
                desired_signal = -BASE_SIZE
            
            # Moderate short: 1d bearish + RSI extreme + below SMA50
            elif trend_1d_bearish and rsi_extreme_high and below_sma50:
                desired_signal = -REDUCED_SIZE
            
            # Rally short: 4h bearish + RSI overbought + below SMA200
            elif trend_4h_bearish and rsi_overbought and below_sma200 and vol_selling:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish and RSI not overbought
                if trend_4h_bullish and rsi_30m[i] < 70:
                    desired_signal = BASE_SIZE if trend_1d_bullish else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_30m[i] > 30:
                    desired_signal = -BASE_SIZE if trend_1d_bearish else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses and RSI overbought
            if trend_1d_bearish and rsi_30m[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses and RSI oversold
            if trend_1d_bullish and rsi_30m[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals