#!/usr/bin/env python3
"""
Experiment #219: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After 218 experiments, static indicators (EMA, HMA) fail in mixed regimes.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - smooth in noise,
fast in trends. Combined with ADX trend strength and Choppiness Index regime detection,
this should outperform static MA strategies.

Key innovations:
1. KAMA(10) adapts smoothing based on volatility (ER = Efficiency Ratio)
2. CHOP(14) regime filter: <38.2 = trend, >61.8 = range (switch logic)
3. ADX(14) confirms trend strength (>25 = valid trend)
4. 1d HTF KAMA for major trend bias (never fight daily trend)
5. ATR(14) 2.5x trailing stop for risk management

Why 4h timeframe:
- 20-50 trades/year target (matches cost model)
- Enough bars for regime detection (CHOP needs 14 periods)
- Proven in exp#214 (Donchian+HMA+RSI got +16.1% return)

Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target: 25-45 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_chop_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    ER (Efficiency Ratio) = |Net Change| / Sum of Absolute Changes
    SC (Smoothing Constant) = [ER * (fast SC - slow SC) + slow SC]^2
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    net_change = np.abs(close_s.diff(er_period))
    sum_changes = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    er = net_change / sum_changes.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    Values: >61.8 = choppy/range, <38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0:
            tr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr_sum += tr
            
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_kama_slope(kama_values, lookback=3):
    """Calculate KAMA slope as percentage change."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        if kama_values[i - lookback] != 0 and not np.isnan(kama_values[i - lookback]):
            slope[i] = (kama_values[i] - kama_values[i - lookback]) / kama_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_slope = calculate_kama_slope(kama_1d, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    kama_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # 4h KAMA for local trend
    kama_4h = calculate_kama(close, er_period=10)
    kama_4h_slope = calculate_kama_slope(kama_4h, 3)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(kama_4h[i]) or np.isnan(kama_4h_slope[i]):
            continue
        
        # === HTF TREND BIAS (1d) ===
        daily_bullish = kama_1d_slope_aligned[i] > 0.15
        daily_bearish = kama_1d_slope_aligned[i] < -0.15
        daily_neutral = not daily_bullish and not daily_bearish
        
        price_above_1d_kama = close[i] > kama_1d_aligned[i]
        price_below_1d_kama = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND (4h KAMA) ===
        hourly_bullish = kama_4h_slope[i] > 0.2
        hourly_bearish = kama_4h_slope[i] < -0.2
        
        price_above_4h_kama = close[i] > kama_4h[i]
        price_below_4h_kama = close[i] < kama_4h[i]
        
        # KAMA crossover signals
        kama_cross_long = False
        kama_cross_short = False
        if i > 0 and not np.isnan(kama_4h[i-1]):
            # Price crossed above KAMA
            if close[i-1] <= kama_4h[i-1] and close[i] > kama_4h[i]:
                kama_cross_long = True
            # Price crossed below KAMA
            if close[i-1] >= kama_4h[i-1] and close[i] < kama_4h[i]:
                kama_cross_short = True
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_14[i] > 25
        trend_very_strong = adx_14[i] > 30
        plus_dominant = plus_di[i] > minus_di[i]
        minus_dominant = minus_di[i] > plus_di[i]
        
        # === REGIME (CHOPPINNESS) ===
        choppy_regime = chop_14[i] > 61.8
        trending_regime = chop_14[i] < 38.2
        neutral_regime = not choppy_regime and not trending_regime
        
        # === MOMENTUM (RSI) ===
        rsi_bullish = rsi_14[i] > 52
        rsi_bearish = rsi_14[i] < 48
        rsi_strong_bull = rsi_14[i] > 58
        rsi_strong_bear = rsi_14[i] < 42
        rsi_extreme_bull = rsi_14[i] > 70
        rsi_extreme_bear = rsi_14[i] < 30
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency
        long_score = 0
        
        # Path 1: Trending regime + ADX strong + KAMA bullish + Daily bullish (strong trend)
        if trending_regime and trend_strong and hourly_bullish and daily_bullish:
            long_score += 4
        
        # Path 2: KAMA cross long + ADX strong + Daily bullish
        if kama_cross_long and trend_strong and daily_bullish:
            long_score += 4
        
        # Path 3: Trending regime + Price above KAMA + RSI bullish + Daily neutral/bullish
        if trending_regime and price_above_4h_kama and rsi_bullish and not daily_bearish:
            long_score += 3
        
        # Path 4: ADX strong + Plus DI dominant + Price above KAMA
        if trend_strong and plus_dominant and price_above_4h_kama:
            long_score += 3
        
        # Path 5: Daily bullish + KAMA cross + RSI confirmation
        if daily_bullish and kama_cross_long and rsi_bullish:
            long_score += 3
        
        # Path 6: Neutral regime + RSI pullback + Daily bullish (mean revert in trend)
        if neutral_regime and rsi_14[i] > 45 and rsi_14[i] < 55 and daily_bullish and price_above_4h_kama:
            long_score += 2
        
        # Path 7: Simple KAMA cross with ADX confirmation (looser)
        if kama_cross_long and adx_14[i] > 20:
            long_score += 2
        
        # Path 8: Choppy regime mean reversion (RSI extreme + Daily bullish)
        if choppy_regime and rsi_extreme_bear and daily_bullish and bars_since_last_trade > 20:
            long_score += 2
        
        # Path 9: Price above KAMA + RSI strong + bars cooldown
        if price_above_4h_kama and rsi_strong_bull and bars_since_last_trade > 30:
            long_score += 1
        
        if long_score >= 4:
            new_signal = current_size
        elif long_score == 3 and bars_since_last_trade > 15:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 25:
            new_signal = current_size * 0.6
        elif long_score >= 1 and bars_since_last_trade > 40:
            new_signal = current_size * 0.4
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Trending regime + ADX strong + KAMA bearish + Daily bearish
        if trending_regime and trend_strong and hourly_bearish and daily_bearish:
            short_score += 4
        
        # Path 2: KAMA cross short + ADX strong + Daily bearish
        if kama_cross_short and trend_strong and daily_bearish:
            short_score += 4
        
        # Path 3: Trending regime + Price below KAMA + RSI bearish + Daily neutral/bearish
        if trending_regime and price_below_4h_kama and rsi_bearish and not daily_bullish:
            short_score += 3
        
        # Path 4: ADX strong + Minus DI dominant + Price below KAMA
        if trend_strong and minus_dominant and price_below_4h_kama:
            short_score += 3
        
        # Path 5: Daily bearish + KAMA cross + RSI confirmation
        if daily_bearish and kama_cross_short and rsi_bearish:
            short_score += 3
        
        # Path 6: Neutral regime + RSI pullback + Daily bearish
        if neutral_regime and rsi_14[i] > 45 and rsi_14[i] < 55 and daily_bearish and price_below_4h_kama:
            short_score += 2
        
        # Path 7: Simple KAMA cross with ADX confirmation (looser)
        if kama_cross_short and adx_14[i] > 20:
            short_score += 2
        
        # Path 8: Choppy regime mean reversion (RSI extreme + Daily bearish)
        if choppy_regime and rsi_extreme_bull and daily_bearish and bars_since_last_trade > 20:
            short_score += 2
        
        # Path 9: Price below KAMA + RSI strong + bars cooldown
        if price_below_4h_kama and rsi_strong_bear and bars_since_last_trade > 30:
            short_score += 1
        
        if short_score >= 4:
            new_signal = -current_size
        elif short_score == 3 and bars_since_last_trade > 15:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 25:
            new_signal = -current_size * 0.6
        elif short_score >= 1 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.4
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 80 bars (~13 days on 4h)
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if daily_bullish and price_above_4h_kama and rsi_14[i] > 50:
                new_signal = current_size * 0.35
            elif daily_bearish and price_below_4h_kama and rsi_14[i] < 50:
                new_signal = -current_size * 0.35
            elif rsi_14[i] > 65 and price_above_1d_kama:
                new_signal = current_size * 0.25
            elif rsi_14[i] < 35 and price_below_1d_kama:
                new_signal = -current_size * 0.25
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but daily turns strongly bearish
            if position_side > 0 and daily_bearish and price_below_1d_kama:
                trend_reversal = True
            # Short position but daily turns strongly bullish
            if position_side < 0 and daily_bullish and price_above_1d_kama:
                trend_reversal = True
        
        # === REGIME CHANGE EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # In trend position but regime becomes choppy
            if position_side > 0 and trending_regime == False and choppy_regime and rsi_14[i] > 65:
                regime_exit = True
            if position_side < 0 and trending_regime == False and choppy_regime and rsi_14[i] < 35:
                regime_exit = True
        
        if stoploss_triggered or trend_reversal or regime_exit:
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