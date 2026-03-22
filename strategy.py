#!/usr/bin/env python3
"""
Experiment #038: 30m HTF Confluence with Regime-Adaptive Entries

Hypothesis: Previous 30m strategies failed (0 trades) due to too many conflicting filters.
This strategy uses LOOSER thresholds while maintaining confluence:

1. 1d HMA(21) for PRIMARY trend bias (slow, reliable direction filter)
2. 4h Choppiness(14) for regime detection (optional enhancement, not mandatory)
3. 30m RSI(14) for entry timing with LOOSE thresholds (30/70 not 20/80)
4. Session filter: 8-20 UTC only (reduces low-liquidity trades)
5. Volume filter: >0.8x 20-bar average (confirms participation)
6. ATR(14) trailing stoploss at 2.5x

Key differences from failed #028:
- LOOSER RSI thresholds (30/70 vs 25/75) to ensure trades
- Regime filter is OPTIONAL not mandatory (either regime OR trend works)
- Smaller position size (0.20-0.25) for 30m fee sensitivity
- Minimum trade frequency safeguard (force entry after 60 bars flat)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Target trades: 40-80/year (1 every 4-9 days on 30m)
Position sizing: 0.20 base, max 0.25 (smaller for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adaptive_4h1d_confluence_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    chop = chop.fillna(50).values
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

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
    
    # Calculate 1d indicators (primary trend bias)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators (regime filter)
    chop_4h_14 = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        14
    )
    chop_4h_14_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_30m_50 = calculate_hma(close, 50)
    vol_sma_20 = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, smaller for 30m)
    BASE_SIZE = 0.20
    MAX_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D TREND BIAS (primary filter) ===
        trend_bullish = close[i] > hma_1d_21_aligned[i]
        trend_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H CHOPPINNESS REGIME (optional enhancement) ===
        choppy_market = chop_4h_14_aligned[i] > 55.0  # Looser threshold
        trending_market = chop_4h_14_aligned[i] < 45.0  # Looser threshold
        
        # === 30M RSI SIGNALS (loose thresholds for trades) ===
        rsi_oversold = rsi_14[i] < 35  # Much looser than 20
        rsi_overbought = rsi_14[i] > 65  # Much looser than 80
        rsi_extreme_oversold = rsi_14[i] < 25
        rsi_extreme_overbought = rsi_14[i] > 75
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_sma_20[i] if not np.isnan(vol_sma_20[i]) else True
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        session_ok = 8 <= utc_hour <= 20
        
        # === POSITION SIZING ===
        if i > 100:
            atr_recent = atr_14[max(0, i-100):i]
            atr_median = np.nanmedian(atr_recent)
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.8, 1.2)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.15, MAX_SIZE)
        
        # === ENTRY LOGIC (multiple paths to ensure trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # PATH 1: Trend-following entry (works in trending market)
        if trend_bullish and trending_market:
            if rsi_oversold and vol_ok and session_ok:
                new_signal = current_size
        elif trend_bearish and trending_market:
            if rsi_overbought and vol_ok and session_ok:
                new_signal = -current_size
        
        # PATH 2: Mean reversion entry (works in choppy market)
        if not in_position:
            if choppy_market:
                if rsi_extreme_oversold and vol_ok:
                    new_signal = current_size * 0.8  # Smaller size for mean reversion
                elif rsi_extreme_overbought and vol_ok:
                    new_signal = -current_size * 0.8
        
        # PATH 3: Simple trend pullback (fallback if regime unclear)
        if new_signal == 0.0 and not in_position:
            if trend_bullish and rsi_14[i] < 40 and close[i] > hma_30m_50[i]:
                if vol_ok:
                    new_signal = current_size * 0.7
            elif trend_bearish and rsi_14[i] > 60 and close[i] < hma_30m_50[i]:
                if vol_ok:
                    new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD (prevent 0 trades) ===
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            # Force entry with weaker signal after long flat period
            if trend_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.5
            elif trend_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.5
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_bearish:
                # Only exit if strong reversal (RSI confirms)
                if rsi_14[i] > 55:
                    trend_reversal = True
            if position_side < 0 and trend_bullish:
                if rsi_14[i] < 45:
                    trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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