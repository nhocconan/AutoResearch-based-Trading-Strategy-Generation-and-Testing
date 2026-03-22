#!/usr/bin/env python3
"""
Experiment #324: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Connors RSI + Choppiness Regime

Hypothesis: 4h timeframe with adaptive KAMA trend + regime-switching logic outperforms fixed HMA strategies.
Key innovations:
1. KAMA (Kaufman Adaptive) adapts to volatility - smooth in chop, responsive in trends
2. 12h KAMA(21) provides major trend direction without excessive lag
3. Choppiness Index switches between trend-follow and mean-revert modes
4. Connors RSI (CRSI) for precise mean-reversion entries in choppy regimes
5. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto bias)
6. Target: 25-40 trades/year on 4h (appropriate frequency, low fee drag)

Why this might beat current best (Sharpe=0.424):
- KAMA outperforms HMA in mixed regimes (2022 crash, 2023-24 range, 2025 bear)
- Regime-switching: trend-follow when CHOP<45, mean-revert when CHOP>55
- Connors RSI catches reversals better than standard RSI
- Looser entry conditions ensure 25+ trades/year across all symbols
- 12h HTF filter prevents counter-trend trades in strong trends

Position sizing: 0.25 base, 0.30 strong conviction (longs), 0.20 (shorts)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_connors_chop_12h1d_asym_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise via Efficiency Ratio.
    """
    n = period
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if i >= n:
            price_change = np.abs(close[i] - close[i-n])
            noise = np.sum(np.abs(np.diff(close[i-n:i+1])))
            
            if noise > 0:
                er = price_change / noise
            else:
                er = 0.0
            
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = np.mean(close[max(0, i-n+1):i+1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values < 10 = oversold, > 90 = overbought
    Better for mean reversion than standard RSI.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        total = streak_period
        if total > 0:
            streak_rsi[i] = (up_streaks / total) * 100.0
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        if len(window) >= rank_period:
            rank = np.sum(window[:-1] < window[-1]) / (len(window) - 1)
            percent_rank[i] = rank * 100.0
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators (major trend direction)
    kama_12h_21 = calculate_kama(df_12h['close'].values, period=21)
    kama_12h_21_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_21)
    
    # Calculate 1d HTF indicators (regime filter)
    kama_1d_21 = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    kama_4h_10 = calculate_kama(close, period=10)
    kama_4h_21 = calculate_kama(close, period=21)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_12h_21_aligned[i]) or np.isnan(kama_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(kama_4h_10[i]) or np.isnan(kama_4h_21[i]):
            continue
        
        # === 12H MAJOR TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > kama_12h_21_aligned[i]
        regime_bear = close[i] < kama_12h_21_aligned[i]
        
        # === 1D REGIME CONFIRMATION (stronger filter) ===
        daily_bull = close[i] > kama_1d_21_aligned[i]
        daily_bear = close[i] < kama_1d_21_aligned[i]
        
        # Strong bull: both 12h and 1d bullish
        strong_bull = regime_bull and daily_bull
        # Strong bear: both 12h and 1d bearish
        strong_bear = regime_bear and daily_bear
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        neutral_chop = 45.0 <= chop_14[i] <= 55.0
        
        # === VOLATILITY REGIME ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 4H LOCAL TREND ===
        kama_bullish = kama_4h_10[i] > kama_4h_21[i]
        kama_bearish = kama_4h_10[i] < kama_4h_21[i]
        
        # KAMA slope
        kama_slope_up = kama_4h_21[i] > kama_4h_21[i-3] if i >= 3 else False
        kama_slope_down = kama_4h_21[i] < kama_4h_21[i-3] if i >= 3 else False
        
        price_above_kama = close[i] > kama_4h_21[i]
        price_below_kama = close[i] < kama_4h_21[i]
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === RSI/CRSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_neutral_oversold = 15.0 <= crsi[i] < 35.0
        crsi_neutral_overbought = 65.0 < crsi[i] <= 85.0
        
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === BOLLINGER BANDS ===
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if bb_mid[i] > 0 else 0
        price_at_bb_lower = close[i] < bb_lower[i] * 1.002
        price_at_bb_upper = close[i] > bb_upper[i] * 0.998
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i] * 0.998
        donchian_breakout_down = close[i] < donchian_lower[i] * 1.002
        
        # === ENTRY LOGIC (REGIME-SWITCHING) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === TRENDING REGIME (CHOP < 45) - Trend Following ===
        if is_trending:
            # LONG: Strong bull regime + KAMA bullish + RSI pullback
            if strong_bull and kama_bullish and kama_slope_up:
                if rsi_14[i] < 55.0 and rsi_rising:
                    new_signal = LONG_BASE * vol_scale
                elif crsi_neutral_oversold and price_above_kama:
                    new_signal = LONG_STRONG * vol_scale
            
            # LONG: Donchian breakout in bull regime
            elif donchian_breakout_up and regime_bull and rsi_14[i] > 45.0:
                new_signal = LONG_BASE * vol_scale
            
            # SHORT: Strong bear regime + KAMA bearish + RSI pullback
            if strong_bear and kama_bearish and kama_slope_down:
                if new_signal == 0.0:
                    if rsi_14[i] > 45.0 and rsi_falling:
                        new_signal = -SHORT_BASE * vol_scale
                    elif crsi_neutral_overbought and price_below_kama:
                        new_signal = -SHORT_STRONG * vol_scale
            
            # SHORT: Donchian breakdown in bear regime
            if donchian_breakout_down and regime_bear and rsi_14[i] < 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
        
        # === CHOPPY REGIME (CHOP > 55) - Mean Reversion ===
        elif is_choppy:
            # LONG: CRSI oversold + price at BB lower + above SMA200
            if crsi_oversold and price_at_bb_lower:
                if price_above_sma200 or regime_bull:
                    new_signal = LONG_BASE * vol_scale
                else:
                    new_signal = LONG_BASE * 0.7 * vol_scale
            
            # LONG: RSI oversold in bull regime
            elif rsi_oversold and regime_bull and price_below_kama:
                new_signal = LONG_BASE * 0.8 * vol_scale
            
            # SHORT: CRSI overbought + price at BB upper
            if crsi_overbought and price_at_bb_upper:
                if new_signal == 0.0:
                    if price_below_sma200 or regime_bear:
                        new_signal = -SHORT_BASE * vol_scale
                    else:
                        new_signal = -SHORT_BASE * 0.7 * vol_scale
            
            # SHORT: RSI overbought in bear regime
            if rsi_overbought and regime_bear and price_above_kama:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) - Mixed ===
        elif neutral_chop:
            # Favor trend direction but with reduced size
            if regime_bull and kama_bullish and rsi_14[i] < 50.0:
                new_signal = LONG_BASE * 0.7 * vol_scale
            elif regime_bear and kama_bearish and rsi_14[i] > 50.0:
                new_signal = -SHORT_BASE * 0.7 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 40 bars (~7 days on 4h)
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if strong_bull and rsi_14[i] > 45.0 and price_above_kama:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif strong_bear and rsi_14[i] < 55.0 and price_below_kama:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
            elif crsi_oversold and regime_bull:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif crsi_overbought and regime_bear:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
        
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
        
        # === RSI/CRSI REVERSAL EXIT ===
        exit_signal = False
        if in_position and position_side != 0:
            # Long: exit on CRSI overbought or RSI > 70
            if position_side > 0 and (crsi_overbought or rsi_14[i] > 70.0):
                exit_signal = True
            # Short: exit on CRSI oversold or RSI < 30
            if position_side < 0 and (crsi_oversold or rsi_14[i] < 30.0):
                exit_signal = True
        
        # === KAMA REVERSAL EXIT ===
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and price_below_kama:
                exit_signal = True
            if position_side < 0 and kama_bullish and price_above_kama:
                exit_signal = True
        
        # === REGIME REVERSAL EXIT ===
        if in_position and position_side != 0:
            if position_side > 0 and strong_bear and price_below_kama:
                exit_signal = True
            if position_side < 0 and strong_bull and price_above_kama:
                exit_signal = True
        
        if stoploss_triggered or exit_signal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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