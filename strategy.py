#!/usr/bin/env python3
"""
Experiment #247: 15m Regime-Adaptive Strategy with CHOP + 4h HMA + 1h RSI + Volume

Hypothesis: 15m timeframe is noisy but can work with strong regime detection.
Using Choppiness Index (CHOP) to detect trend vs range regimes, then applying
different logic per regime:
- Trending (CHOP < 38): Follow 4h HMA trend with 1h RSI pullback entries
- Ranging (CHOP > 61): Mean revert at Bollinger Band extremes
- Neutral (38-61): Stay flat or reduce position size

Why this might work on 15m:
- 15m captures intraday swings that 1h/4h miss
- CHOP regime filter prevents trend strategies in choppy markets (major failure cause)
- 4h HMA provides strong macro bias (proven in best strategies)
- 1h RSI provides entry timing without look-ahead
- Volume confirmation filters false breakouts
- Conservative sizing (0.25) + ATR stoploss controls drawdown

Key improvements over failed 15m experiments (#235, #241):
- Added CHOP regime detection (was missing in failed strategies)
- Different logic per regime instead of one-size-fits-all
- Volume confirmation on breakouts
- Looser RSI thresholds (25/75 vs 20/80) for more trades
- 1h RSI aligned properly via mtf_data helper

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (max 0.30)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_chop_regime_4h_hma_1h_rsi_volume_atr_v1"
timeframe = "15m"
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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100
    
    return upper.values, lower.values, sma.values, bb_width.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    chop[:period] = 50  # neutral for initial bars
    
    return chop

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average for volume confirmation."""
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.12
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (4h HMA) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h RSI FOR ENTRY TIMING ===
        rsi_1h_val = rsi_1h_aligned[i]
        rsi_1h_bullish = rsi_1h_val > 40 and rsi_1h_val < 70  # Pullback zone
        rsi_1h_bearish = rsi_1h_val < 60 and rsi_1h_val > 30  # Rally zone
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61 = Ranging (mean reversion)
        # CHOP < 38 = Trending (trend following)
        # 38-61 = Neutral (reduced size or flat)
        is_trending = chop[i] < 40
        is_ranging = chop[i] > 60
        is_neutral = not is_trending and not is_ranging
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]  # 20% above average
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- TRENDING REGIME (CHOP < 40) ---
        if is_trending:
            # Long: 4h bullish + 1h RSI pullback + volume confirmation
            if bull_trend_4h and rsi_1h_bullish:
                if rsi_7[i] < 75 and volume_confirmed:
                    new_signal = SIZE_BASE
            
            # Short: 4h bearish + 1h RSI rally + volume confirmation
            if bear_trend_4h and rsi_1h_bearish:
                if rsi_7[i] > 25 and volume_confirmed:
                    new_signal = -SIZE_BASE
        
        # --- RANGING REGIME (CHOP > 60) ---
        if is_ranging:
            # Long: Price at BB lower + RSI oversold
            if close[i] < bb_lower[i] and rsi_14[i] < 30:
                new_signal = SIZE_BASE
            
            # Short: Price at BB upper + RSI overbought
            if close[i] > bb_upper[i] and rsi_14[i] > 70:
                new_signal = -SIZE_BASE
        
        # --- NEUTRAL REGIME (38-61) ---
        if is_neutral:
            # Only enter on strong signals, reduced size
            # Long: Strong 4h bullish + RSI very oversold
            if bull_trend_4h and rsi_14[i] < 25:
                new_signal = SIZE_HALF
            
            # Short: Strong 4h bearish + RSI very overbought
            if bear_trend_4h and rsi_14[i] > 75:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals