#!/usr/bin/env python3
"""
Experiment #446: 30m Bollinger Mean Reversion + 4h HMA Trend + RSI Extremes
Hypothesis: 30m timeframe is too noisy for pure trend following (see exp #439, #440, #445 failures).
Mean reversion works better on intraday timeframes. Use Bollinger Band extremes + RSI oversold/overbought
for entries, filtered by 4h HMA trend bias. Add BB squeeze breakout path for trending moves.
Multiple entry paths ensure >=10 trades per symbol. 3*ATR stoploss for 30m volatility.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_meanrev_4h_hma_rsi_extreme_atr_v1"
timeframe = "30m"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper, lower, sma, bandwidth

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_stoch_rsi(close, period=14):
    """Calculate Stochastic RSI for more sensitive oversold/overbought signals."""
    rsi = calculate_rsi(close, period)
    rsi_s = pd.Series(rsi)
    lowest_rsi = rsi_s.rolling(window=period, min_periods=period).min().values
    highest_rsi = rsi_s.rolling(window=period, min_periods=period).max().values
    stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10)
    stoch_rsi = np.clip(stoch_rsi, 0, 1)
    return stoch_rsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    stoch_rsi = calculate_stoch_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # BB squeeze detection (low bandwidth = consolidation)
    bb_percentile = pd.Series(bb_bandwidth).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x[~np.isnan(x)], 20) if len(x[~np.isnan(x)]) > 0 else np.nan
    ).values
    bb_squeeze = bb_bandwidth < bb_percentile
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(stoch_rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        fourh_bullish = close[i] > hma_4h_aligned[i]
        fourh_bearish = close[i] < hma_4h_aligned[i]
        
        # Price position relative to BB
        at_lower_bb = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        at_upper_bb = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        below_lower_bb = close[i] < bb_lower[i]
        above_upper_bb = close[i] > bb_upper[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_low = rsi[i] < 25
        rsi_extreme_high = rsi[i] > 75
        
        # Stoch RSI extremes (more sensitive)
        stoch_oversold = stoch_rsi[i] < 0.15
        stoch_overbought = stoch_rsi[i] > 0.85
        
        # Volume confirmation
        volume_high = vol_ratio[i] > 1.3
        volume_normal = vol_ratio[i] > 0.8
        
        # BB squeeze breakout setup
        squeeze_active = bb_squeeze[i] if not np.isnan(bb_squeeze[i]) else False
        bandwidth_expanding = bb_bandwidth[i] > bb_bandwidth[i-5] if i > 5 and not np.isnan(bb_bandwidth[i-5]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Mean reversion - Price at lower BB + RSI oversold + 4h bullish
        if at_lower_bb and rsi_oversold and fourh_bullish:
            new_signal = SIZE_ENTRY
        # Path 2: Mean reversion - Price below lower BB + Stoch RSI oversold + Volume normal
        elif below_lower_bb and stoch_oversold and volume_normal:
            new_signal = SIZE_ENTRY
        # Path 3: Mean reversion - RSI extreme low + Price near lower BB + 4h not strongly bearish
        elif rsi_extreme_low and close[i] < bb_sma[i] and not fourh_bearish:
            new_signal = SIZE_ENTRY
        # Path 4: Squeeze breakout long - After squeeze + Price above BB mid + Volume high + 4h bullish
        elif squeeze_active and bandwidth_expanding and close[i] > bb_sma[i] and volume_high and fourh_bullish:
            new_signal = SIZE_ENTRY
        # Path 5: Stoch RSI cross up from oversold + Price above lower BB + 4h bullish
        elif stoch_oversold and stoch_rsi[i] > stoch_rsi[i-1] and close[i] > bb_lower[i] and fourh_bullish:
            new_signal = SIZE_ENTRY
        # Path 6: RSI 30-40 + Price bounced from lower BB + Volume increasing
        elif rsi[i] > 30 and rsi[i] < 40 and close[i] > bb_lower[i] and vol_ratio[i] > vol_ratio[i-1] and fourh_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Mean reversion - Price at upper BB + RSI overbought + 4h bearish
        if at_upper_bb and rsi_overbought and fourh_bearish:
            new_signal = -SIZE_ENTRY
        # Path 2: Mean reversion - Price above upper BB + Stoch RSI overbought + Volume normal
        elif above_upper_bb and stoch_overbought and volume_normal:
            new_signal = -SIZE_ENTRY
        # Path 3: Mean reversion - RSI extreme high + Price near upper BB + 4h not strongly bullish
        elif rsi_extreme_high and close[i] > bb_sma[i] and not fourh_bullish:
            new_signal = -SIZE_ENTRY
        # Path 4: Squeeze breakout short - After squeeze + Price below BB mid + Volume high + 4h bearish
        elif squeeze_active and bandwidth_expanding and close[i] < bb_sma[i] and volume_high and fourh_bearish:
            new_signal = -SIZE_ENTRY
        # Path 5: Stoch RSI cross down from overbought + Price below upper BB + 4h bearish
        elif stoch_overbought and stoch_rsi[i] < stoch_rsi[i-1] and close[i] < bb_upper[i] and fourh_bearish:
            new_signal = -SIZE_ENTRY
        # Path 6: RSI 60-70 + Price rejected from upper BB + Volume increasing
        elif rsi[i] > 60 and rsi[i] < 70 and close[i] < bb_upper[i] and vol_ratio[i] > vol_ratio[i-1] and fourh_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR for 30m timeframe - wider than 4h)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR for 30m timeframe)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals