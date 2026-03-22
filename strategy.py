#!/usr/bin/env python3
"""
Experiment #030: 1h Multi-Timeframe Regime-Adaptive with Session Filter

Hypothesis: 1h strategies fail due to too many trades (fee drag) or too few (no stats).
This uses 4h/12h HMA for TREND DIRECTION, 1h only for ENTRY TIMING.
Key innovation: Choppiness Index regime detection + Connors RSI + session filter.

Logic:
1. 4h HMA(21) > HMA(50) = bullish bias (only look for longs)
2. 12h HMA(21) confirms major trend (price above for long bias)
3. Choppiness(14): >55 = range (mean revert), <45 = trend (breakout)
4. Connors RSI <20 (long) or >80 (short) for entry timing
5. Session filter: only 8-20 UTC (high liquidity hours)
6. Volume > 0.8x 20-bar average
7. ATR(14) trailing stoploss at 2.5x

Why this works:
- HTF filters reduce whipsaw (4h/12h trend = fewer false signals)
- Session filter avoids low-liquidity hours (less slippage)
- Choppiness adapts to market regime (trend vs range)
- Connors RSI catches pullbacks in trend (better R:R than breakout)

Timeframe: 1h (REQUIRED)
HTF: 4h, 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 (conservative for lower TF)
Target trades: 40-80/year (strict confluence = fewer but higher quality)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_connors_4h_12h_hma_session_vol_atr_v1"
timeframe = "1h"
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
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range == 0, 1e-10, price_range)  # avoid div by zero
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) component
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi_close = 100 - (100 / (1 + rs))
    
    # Streak RSI(2) component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / avg_streak_loss
    streak_rs = streak_rs.replace([np.inf, -np.inf], np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # PercentRank(100) component
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    ) * 100
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    return crsi.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4H indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Calculate 12H indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume SMA for filter
    volume_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100  # Track last trade for frequency control
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(volume_sma20[i]) or volume_sma20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma20[i]
        
        # === 4H TREND BIAS ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 12H MAJOR TREND CONFIRMATION ===
        trend_12h_bullish = close[i] > hma_12h_21_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 55  # Range/choppy market
        chop_trend = chop_14[i] < 45  # Trending market
        
        # === CONNORS RSI ENTRY SIGNALS ===
        crsi_oversold = crsi[i] < 20  # Long entry
        crsi_overbought = crsi[i] > 80  # Short entry
        
        # === RSI CONFIRMATION ===
        rsi_ok_long = rsi_14[i] < 50  # Not overbought for long
        rsi_ok_short = rsi_14[i] > 50  # Not oversold for short
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.2)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.30)  # Keep in safe range
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need 4h trend + 12h bias + CRSI oversold + session + volume
        long_confluence = 0
        if hma_4h_bullish:
            long_confluence += 1
        if trend_12h_bullish:
            long_confluence += 1
        if crsi_oversold:
            long_confluence += 1
        if in_session:
            long_confluence += 0.5
        if volume_ok:
            long_confluence += 0.5
        if rsi_ok_long:
            long_confluence += 0.5
        
        # Enter long if confluence >= 3.5 (need at least 3 strong signals)
        if long_confluence >= 3.5 and hma_4h_bullish and trend_12h_bullish and crsi_oversold:
            new_signal = current_size
        
        # SHORT ENTRY: Need 4h trend + 12h bias + CRSI overbought + session + volume
        short_confluence = 0
        if hma_4h_bearish:
            short_confluence += 1
        if trend_12h_bearish:
            short_confluence += 1
        if crsi_overbought:
            short_confluence += 1
        if in_session:
            short_confluence += 0.5
        if volume_ok:
            short_confluence += 0.5
        if rsi_ok_short:
            short_confluence += 0.5
        
        # Enter short if confluence >= 3.5
        if short_confluence >= 3.5 and hma_4h_bearish and trend_12h_bearish and crsi_overbought:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~4 days on 1h), slightly loosen entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and trend_12h_bullish and crsi[i] < 30 and in_session:
                new_signal = current_size * 0.7
            elif hma_4h_bearish and trend_12h_bearish and crsi[i] > 70 and in_session:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h HMA turns bearish
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            # Exit short if 4h HMA turns bullish
            if position_side < 0 and hma_4h_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT (take profit) ===
        take_profit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 75:
                take_profit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 25:
                take_profit = True
        
        # Apply stoploss or trend reversal or take profit
        if stoploss_triggered or trend_reversal or take_profit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals