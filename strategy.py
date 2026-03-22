#!/usr/bin/env python3
"""
Experiment #078: 30m Primary + 4h/1d HTF — Regime-Adaptive Connors RSI

Hypothesis: 30m timeframe can work IF we use strict confluence filters to limit
trades to 30-80/year. Previous 30m strategies failed due to too many trades (>200/yr)
causing fee drag. This strategy uses:

1. 4h HMA(21) for intermediate trend direction (only trade WITH 4h trend)
2. 1d HMA slope for major regime bias (bullish/bearish market context)
3. 30m Choppiness(14) for market state (range vs trend)
4. 30m Connors RSI for entry timing (extreme mean reversion signals)
5. Session filter: 8-20 UTC only (highest volume hours, avoid Asia low-volume)
6. Volume filter: volume > 0.8x 20-bar average (confirm participation)
7. ATR(14) stoploss: 2.5x trailing stop
8. Position size: 0.25 (smaller for 30m due to higher trade frequency)

Why this should work:
- 4h/1d HTF ensures we trade with major trend (prevents counter-trend disasters)
- Choppiness regime detection adapts entry thresholds (range=extreme CRSI, trend=moderate)
- Session filter cuts 50%+ of trades (low-volume hours have false breakouts)
- Volume filter confirms real participation (avoids fake moves)
- Connors RSI has proven 75% win rate for mean reversion entries
- 3+ confluence filters naturally limit to 40-80 trades/year

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_connors_hma_4h1d_session_v1"
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
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)  # avoid div by zero
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Component 3: Percent Rank of price change
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to hours
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
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 30m HMA for local trend
    hma_30m_21 = calculate_hma(close, 21)
    hma_30m_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, smaller for 30m)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # Skip entries outside high-volume session
        if in_session == False and not in_position:
            signals[i] = 0.0
            continue
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 1D TREND BIAS (MAJOR) ===
        # HMA slope > 0.5 = bullish bias (prefer longs)
        # HMA slope < -0.5 = bearish bias (prefer shorts)
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range market (mean revert strategy)
        # CHOP < 45 = trend market (trend follow strategy)
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        # Range market: extreme CRSI for mean reversion
        # Trend market: moderate CRSI for pullback entries
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_moderate_low = crsi[i] < 30
        crsi_moderate_high = crsi[i] > 70
        
        # === 30M LOCAL TREND ===
        hma_30m_bullish = hma_30m_21[i] > hma_30m_50[i]
        hma_30m_bearish = hma_30m_21[i] < hma_30m_50[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets or weak volume
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.6
        if not volume_ok:
            current_size = current_size * 0.7
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_confluence = 0
        
        # Confluence 1: 4h trend bullish OR 1d trend bullish
        if hma_4h_bullish or trend_1d_bullish:
            long_confluence += 1
        
        # Confluence 2: Price above 1d HMA
        if price_above_1d_hma:
            long_confluence += 1
        
        # Confluence 3: Volume OK
        if volume_ok:
            long_confluence += 1
        
        # Confluence 4: In session
        if in_session:
            long_confluence += 1
        
        # Confluence 5: CRSI signal
        if is_range_market and crsi_oversold:
            long_confluence += 2  # Stronger signal in range
        elif is_trend_market and crsi_moderate_low:
            long_confluence += 1
        
        # Need at least 4 confluence for long entry
        if long_confluence >= 4:
            if is_range_market and crsi_oversold:
                new_signal = current_size
            elif is_trend_market and crsi_moderate_low and hma_4h_bullish:
                new_signal = current_size
            elif trend_1d_bullish and crsi[i] < 25:
                new_signal = current_size * 0.8
        
        # SHORT ENTRIES
        short_confluence = 0
        
        # Confluence 1: 4h trend bearish OR 1d trend bearish
        if hma_4h_bearish or trend_1d_bearish:
            short_confluence += 1
        
        # Confluence 2: Price below 1d HMA
        if price_below_1d_hma:
            short_confluence += 1
        
        # Confluence 3: Volume OK
        if volume_ok:
            short_confluence += 1
        
        # Confluence 4: In session
        if in_session:
            short_confluence += 1
        
        # Confluence 5: CRSI signal
        if is_range_market and crsi_overbought:
            short_confluence += 2  # Stronger signal in range
        elif is_trend_market and crsi_moderate_high:
            short_confluence += 1
        
        # Need at least 4 confluence for short entry
        if short_confluence >= 4:
            if is_range_market and crsi_overbought:
                new_signal = -current_size
            elif is_trend_market and crsi_moderate_high and hma_4h_bearish:
                new_signal = -current_size
            elif trend_1d_bearish and crsi[i] > 75:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~4 days on 30m), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and hma_4h_bullish and crsi[i] < 35:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and hma_4h_bearish and crsi[i] > 65:
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if major trend reverses against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and trend_1d_bearish and hma_4h_bearish:
                regime_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and trend_1d_bullish and hma_4h_bullish:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
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