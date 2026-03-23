#!/usr/bin/env python3
"""
Experiment #908: 30m Primary + 4h/1d HTF — Vol Spike Reversion + HTF Trend + Session Filter

Hypothesis: After 600+ failed strategies, the key insight is that BTC/ETH fail on pure trend
following but succeed on VOLATILITY MEAN REVERSION during panic spikes. Research shows:

1. Vol Spike Reversion: ATR(7)/ATR(30) > 2.0 indicates extreme panic → reversion likely
2. Price < BB(20, 2.5) confirms oversold condition during vol spike
3. 4h HMA21 provides trend direction (only trade with HTF trend)
4. 1d HMA21 provides macro bias (avoid counter-macro trades)
5. Session filter (8-20 UTC) avoids low-liquidity Asian session whipsaws
6. Volume > 0.8x average confirms genuine panic not fake breakout

Why 30m with strict filters:
- Lower TF gives better entry timing within HTF trend
- Strict confluence (4+ filters) ensures only 30-80 trades/year
- Vol spike reversion works in both bull AND bear markets (unlike trend following)
- Session filter reduces false signals during low-liquidity periods

Key differences from failed experiments:
- VOL SPIKE focus (not CRSI/Donchian which failed 20+ times)
- Stricter entry confluence (4+ conditions must align)
- Session filter for lower TF (8-20 UTC only)
- Asymmetric sizing: 0.25 for vol spike entries, 0.15 for pullback entries
- Exit on vol normalization (ATR ratio < 1.3) not just price target

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_spike_reversion_4h1d_hma_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_bollinger(close, period=20, std_dev=2.5):
    """Bollinger Bands with configurable std dev."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

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

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_sma(volume, period=20):
    """Volume SMA for volume filter."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_dev=2.5)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Vol spike entries
    PULLBACK_SIZE = 0.15  # Pullback entries
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_vol_ratio = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_ratio = atr_7[i] / atr_30[i]
        vol_spike = vol_ratio > 2.0  # Extreme volatility
        vol_normalizing = vol_ratio < 1.3  # Volatility calming down
        vol_elevated = vol_ratio > 1.5  # Moderately elevated
        
        # === PRICE POSITION vs BOLLINGER ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        near_bb_lower = close[i] < bb_lower[i] * 1.02  # Within 2% of lower band
        
        # === HTF TREND FILTERS ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SHORT-TERM TREND ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_oversold = rsi_14[i] < 25
        rsi_extreme_overbought = rsi_14[i] > 75
        rsi_neutral = 40 < rsi_14[i] < 60
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        volume_spike = volume[i] > 1.5 * vol_sma_20[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === CHOPPINESS REGIME ===
        ranging_regime = chop_14[i] > 55
        trending_regime = chop_14[i] < 45
        
        desired_signal = 0.0
        
        # === VOL SPIKE REVERSION LONG (Primary Entry) ===
        # Requires: vol spike + price below BB + HTF trend bull + session + volume
        if vol_spike and below_bb_lower and macro_bull and volume_ok:
            if in_session or rsi_extreme_oversold:  # Session OR extreme RSI
                desired_signal = BASE_SIZE
        
        # === VOL SPIKE REVERSION SHORT (Primary Entry) ===
        if vol_spike and above_bb_upper and macro_bear and volume_ok:
            if in_session or rsi_extreme_overbought:
                desired_signal = -BASE_SIZE
        
        # === PULLBACK LONG (Secondary Entry in Bull Trend) ===
        # Enter on pullback when vol elevated but not extreme
        if not vol_spike and vol_elevated and macro_bull and trend_4h_bullish:
            if near_bb_lower and rsi_oversold and volume_ok:
                desired_signal = PULLBACK_SIZE
        
        # === PULLBACK SHORT (Secondary Entry in Bear Trend) ===
        if not vol_spike and vol_elevated and macro_bear and trend_4h_bearish:
            if close[i] > bb_upper[i] * 0.98 and rsi_overbought and volume_ok:
                desired_signal = -PULLBACK_SIZE
        
        # === RANGING REGIME MEAN REVERSION ===
        if ranging_regime and not vol_spike:
            # Long at lower BB with RSI confirmation
            if below_bb_lower and rsi_oversold and above_sma200:
                if in_session:
                    desired_signal = REDUCED_SIZE
            
            # Short at upper BB with RSI confirmation
            if above_bb_upper and rsi_overbought and below_sma200:
                if in_session:
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME BREAKOUT ===
        if trending_regime and macro_bull and trend_4h_bullish:
            # Long on pullback to SMA50 in uptrend
            if close[i] < sma_50[i] * 1.01 and close[i] > sma_50[i] * 0.99 and rsi_neutral:
                if in_session and volume_ok:
                    desired_signal = REDUCED_SIZE
        
        if trending_regime and macro_bear and trend_4h_bearish:
            # Short on pullback to SMA50 in downtrend
            if close[i] > sma_50[i] * 0.99 and close[i] < sma_50[i] * 1.01 and rsi_neutral:
                if in_session and volume_ok:
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
        
        # === VOL NORMALIZATION EXIT ===
        # Exit when volatility normalizes (mean reversion complete)
        if in_position and vol_normalizing and not vol_spike:
            if position_side > 0 and close[i] > bb_sma[i]:
                desired_signal = 0.0  # Long reached mean
            if position_side < 0 and close[i] < bb_sma[i]:
                desired_signal = 0.0  # Short reached mean
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro trend intact and not at profit target
                if macro_bull and close[i] < bb_sma[i] * 1.05:
                    desired_signal = BASE_SIZE if vol_spike else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if macro trend intact and not at profit target
                if macro_bear and close[i] > bb_sma[i] * 0.95:
                    desired_signal = -BASE_SIZE if vol_spike else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + 4h trend reverses
            if macro_bear and trend_4h_bearish:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + 4h trend reverses
            if macro_bull and trend_4h_bullish:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_extreme_oversold:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE:
                desired_signal = BASE_SIZE
            elif desired_signal >= PULLBACK_SIZE:
                desired_signal = PULLBACK_SIZE
            else:
                desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -PULLBACK_SIZE:
                desired_signal = -PULLBACK_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_7[i]  # Use short-term ATR for stops
                entry_vol_ratio = vol_ratio
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_7[i]
                entry_vol_ratio = vol_ratio
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
                entry_vol_ratio = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals