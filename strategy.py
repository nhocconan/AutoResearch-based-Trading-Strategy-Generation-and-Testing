#!/usr/bin/env python3
"""
Experiment #108: 30m Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion with Session Filter

Hypothesis: Lower timeframe (30m) strategies fail due to excessive trade frequency and fee drag.
This strategy uses PROVEN elements from #106 (vol-spike + Connors + BB) but adapted for 30m with:

1. 1d HMA(21) SLOPE: Major trend bias (only trade in direction of daily trend)
2. 4h CHOPPINESS INDEX: Regime filter (CHOP>55 = range/mean-revert, CHOP<45 = trend/pullback)
3. 30m CONNORS RSI: Entry timing (CRSI<20 long, CRSI>80 short)
4. SESSION FILTER: Only trade 8-20 UTC (highest liquidity, lowest slippage)
5. VOLUME FILTER: Volume > 0.8x 20-bar average (confirm participation)
6. VOLATILITY SPIKE: ATR(7)/ATR(30) > 1.5 (capitulation events)
7. BOLLINGER BANDS: Price extreme confirms entry (BB 2.5 std)

Why this should work on 30m:
- 1d trend filter prevents counter-trend trades (major failure point)
- Session filter reduces trades by ~60% (only 12h of 24h)
- Volume filter avoids low-liquidity traps
- 3+ confluence required = ~40-70 trades/year target
- Asymmetric: 0.25 size with trend, 0.15 against (but only with extreme CRSI)

Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.15-0.25 discrete (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-70/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_connors_session_4h1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def get_utc_hour(open_time_ms):
    """Extract UTC hour from millisecond timestamp."""
    return (open_time_ms // (1000 * 3600)) % 24

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
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 4h indicators
    chop_4h = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # Calculate 30m indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility spike ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    # Session hours (8-20 UTC)
    session_mask = np.array([8 <= get_utc_hour(t) <= 20 for t in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 30m)
    BASE_SIZE_WITH_TREND = 0.25
    BASE_SIZE_COUNTER = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    # Minimum bars between trades (30m: 48 bars = 1 day, target ~60 trades/year)
    MIN_BARS_BETWEEN_TRADES = 40
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_4h_aligned[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]) or np.isnan(vol_avg[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = session_mask[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        is_range_market = chop_4h_aligned[i] > 55
        is_trend_market = chop_4h_aligned[i] < 45
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.5
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        crsi_extreme_low = crsi[i] < 15
        crsi_extreme_high = crsi[i] > 85
        crsi_moderate_low = crsi[i] < 30
        crsi_moderate_high = crsi[i] > 70
        
        # === BARS SINCE LAST TRADE ===
        bars_since_last_trade = i - last_trade_bar
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        current_size = BASE_SIZE_WITH_TREND
        
        # LONG ENTRIES - Require 3+ confluence
        long_confluence = 0
        long_with_trend = False
        
        # Confluence 1: Session filter (mandatory for 30m)
        if in_session:
            long_confluence += 1
        
        # Confluence 2: Volume confirmation
        if volume_ok:
            long_confluence += 1
        
        # Confluence 3: 1d trend bullish OR CRSI extreme
        if trend_1d_bullish:
            long_confluence += 1
            long_with_trend = True
        elif crsi_extreme_low:
            long_confluence += 1
        
        # Confluence 4: Vol spike OR BB lower
        if vol_spike or price_below_bb_lower:
            long_confluence += 1
        
        # Confluence 5: CRSI oversold
        if crsi_oversold:
            long_confluence += 1
        
        # Confluence 6: Range market (mean revert favorable)
        if is_range_market:
            long_confluence += 1
        
        # Entry: need 4+ confluence, or 3+ with trend
        if long_confluence >= 4 or (long_confluence >= 3 and long_with_trend):
            if bars_since_last_trade > MIN_BARS_BETWEEN_TRADES:
                current_size = BASE_SIZE_WITH_TREND if long_with_trend else BASE_SIZE_COUNTER
                new_signal = current_size
        
        # SHORT ENTRIES
        short_confluence = 0
        short_with_trend = False
        
        # Confluence 1: Session filter
        if in_session:
            short_confluence += 1
        
        # Confluence 2: Volume confirmation
        if volume_ok:
            short_confluence += 1
        
        # Confluence 3: 1d trend bearish OR CRSI extreme
        if trend_1d_bearish:
            short_confluence += 1
            short_with_trend = True
        elif crsi_extreme_high:
            short_confluence += 1
        
        # Confluence 4: Vol spike OR BB upper
        if vol_spike or price_above_bb_upper:
            short_confluence += 1
        
        # Confluence 5: CRSI overbought
        if crsi_overbought:
            short_confluence += 1
        
        # Confluence 6: Range market
        if is_range_market:
            short_confluence += 1
        
        # Entry: need 4+ confluence, or 3+ with trend
        if short_confluence >= 4 or (short_confluence >= 3 and short_with_trend):
            if bars_since_last_trade > MIN_BARS_BETWEEN_TRADES:
                current_size = BASE_SIZE_WITH_TREND if short_with_trend else BASE_SIZE_COUNTER
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~4 days on 30m) and conditions favorable
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if in_session and volume_ok:
                if trend_1d_bullish and crsi_moderate_low:
                    new_signal = BASE_SIZE_COUNTER * 0.8
                elif trend_1d_bearish and crsi_moderate_high:
                    new_signal = -BASE_SIZE_COUNTER * 0.8
                elif crsi_extreme_low:
                    new_signal = BASE_SIZE_COUNTER * 0.6
                elif crsi_extreme_high:
                    new_signal = -BASE_SIZE_COUNTER * 0.6
        
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
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if trend turns bearish strongly
            if position_side > 0 and trend_1d_bearish and hma_1d_slope_aligned[i] < -1.0:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and trend_1d_bullish and hma_1d_slope_aligned[i] > 1.0:
                regime_reversal = True
        
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