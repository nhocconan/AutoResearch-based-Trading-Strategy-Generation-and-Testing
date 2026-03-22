#!/usr/bin/env python3
"""
Experiment #197: 1d Primary + 1w HTF — Dual Regime Adaptive Strategy

Hypothesis: Previous 1d strategies failed because they were either too strict (0 trades)
or used wrong regime detection. This strategy combines:

1. DUAL REGIME: Choppiness Index determines mode (CHOP>55=mean-revert, CHOP<45=trend-follow)
2. MEAN REVERT MODE: Connors RSI extremes (<20/>80) + Bollinger Band breaks
3. TREND MODE: Donchian(20) breakouts + 1w HMA trend confirmation
4. VOLUME CONFIRMATION: Volume > SMA(20) volume on breakout bars
5. ASYMMETRIC SIZING: Larger positions in confirmed trends, smaller in mean-revert
6. 1w HTF BIAS: Weekly HMA slope prevents counter-trend trades in strong moves

Why this should work on 1d:
- 1d timeframe naturally limits trades to 20-50/year (low fee drag)
- Dual regime adapts to market conditions (range vs trend)
- 1w HTF prevents fighting major trends (critical for 2022 crash survival)
- Multiple entry paths ensure minimum trade frequency
- Conservative sizing (0.25-0.30) limits drawdown during crypto crashes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.30 for high-conviction
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dualregime_connors_donchian_1w_v1"
timeframe = "1d"
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
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
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

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for confirmation."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    # 1d HMA for additional trend confirmation
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    TREND_SIZE = 0.30
    MEAN_REVERT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1W TREND BIAS (major trend filter) ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        trend_1w_neutral = not trend_1w_bullish and not trend_1w_bearish
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        trend_1d_bullish = hma_1d_slope[i] > 0.3
        trend_1d_bearish = hma_1d_slope[i] < -0.3
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        is_transition = not is_range_market and not is_trend_market
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma_20[i] * 1.2
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_width = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if bb_mid[i] > 0 else 0
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 15
        crsi_extreme_high = crsi[i] > 85
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === POSITION SIZING BASED ON REGIME ===
        current_size = BASE_SIZE
        if is_trend_market:
            current_size = TREND_SIZE
        elif is_range_market:
            current_size = MEAN_REVERT_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency
        long_score = 0
        long_confidence = 0
        
        # Path 1: Range market + CRSI extreme oversold (mean revert)
        if is_range_market and crsi_extreme_low:
            long_score += 3
            long_confidence += 2
        
        # Path 2: Range market + BB lower + CRSI oversold
        if is_range_market and price_below_bb_lower and crsi_oversold:
            long_score += 3
            long_confidence += 2
        
        # Path 3: Trend market + Donchian breakout + volume + 1w bullish
        if is_trend_market and breakout_long and volume_confirmed and (trend_1w_bullish or trend_1w_neutral):
            long_score += 4
            long_confidence += 3
        
        # Path 4: Trend market + pullback to HMA + 1w bullish bias
        if is_trend_market and trend_1w_bullish and close[i] < hma_1d_21[i] * 1.02 and close[i] > hma_1d_21[i] * 0.98 and crsi[i] < 40:
            long_score += 2
            long_confidence += 1
        
        # Path 5: 1w bullish + CRSI moderate oversold (trend continuation)
        if trend_1w_bullish and crsi[i] < 35 and price_above_1w_hma:
            long_score += 2
            long_confidence += 1
        
        # Path 6: Simple oversold fallback (ensures minimum trades)
        if crsi[i] < 20 and price_below_bb_lower and bars_since_last_trade > 20:
            long_score += 2
            long_confidence += 1
        
        # Determine long signal strength
        if long_score >= 4:
            new_signal = current_size
        elif long_score >= 3:
            new_signal = current_size * 0.8
        elif long_score >= 2 and bars_since_last_trade > 15:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        short_confidence = 0
        
        # Path 1: Range market + CRSI extreme overbought
        if is_range_market and crsi_extreme_high:
            short_score += 3
            short_confidence += 2
        
        # Path 2: Range market + BB upper + CRSI overbought
        if is_range_market and price_above_bb_upper and crsi_overbought:
            short_score += 3
            short_confidence += 2
        
        # Path 3: Trend market + Donchian breakdown + volume + 1w bearish
        if is_trend_market and breakout_short and volume_confirmed and (trend_1w_bearish or trend_1w_neutral):
            short_score += 4
            short_confidence += 3
        
        # Path 4: Trend market + rally to HMA + 1w bearish bias
        if is_trend_market and trend_1w_bearish and close[i] > hma_1d_21[i] * 0.98 and close[i] < hma_1d_21[i] * 1.02 and crsi[i] > 60:
            short_score += 2
            short_confidence += 1
        
        # Path 5: 1w bearish + CRSI moderate overbought
        if trend_1w_bearish and crsi[i] > 65 and price_below_1w_hma:
            short_score += 2
            short_confidence += 1
        
        # Path 6: Simple overbought fallback
        if crsi[i] > 80 and price_above_bb_upper and bars_since_last_trade > 20:
            short_score += 2
            short_confidence += 1
        
        if short_score >= 4:
            new_signal = -current_size
        elif short_score >= 3:
            new_signal = -current_size * 0.8
        elif short_score >= 2 and bars_since_last_trade > 15:
            new_signal = -current_size * 0.6
        
        # === TRADE FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~60 days on 1d)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and crsi[i] < 40:
                new_signal = BASE_SIZE * 0.4
            elif trend_1w_bearish and crsi[i] > 60:
                new_signal = -BASE_SIZE * 0.4
            elif crsi[i] < 25:
                new_signal = BASE_SIZE * 0.35
            elif crsi[i] > 75:
                new_signal = -BASE_SIZE * 0.35
        
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
            # Exit long if regime shifts to strong trend bearish
            if position_side > 0 and is_trend_market and trend_1w_bearish and trend_1d_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong trend bullish
            if position_side < 0 and is_trend_market and trend_1w_bullish and trend_1d_bullish:
                regime_reversal = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 80:
                crsi_reversal = True
            if position_side < 0 and crsi[i] < 20:
                crsi_reversal = True
        
        if stoploss_triggered or regime_reversal or crsi_reversal:
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