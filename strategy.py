#!/usr/bin/env python3
"""
Experiment #181: 4h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + Funding Proxy

Hypothesis: Previous 4h strategies failed because they used simple RSI/trend indicators that
don't capture reversal dynamics in bear/range markets. Research shows Ehlers Fisher Transform
catches reversals better than RSI (75%+ win rate on extremes), and Choppiness Index regime
filter prevents trend-following in chop (where it gets whipsawed).

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, enters at -1.5/+1.5 extremes (proven reversal indicator)
2. CHOPPINESS INDEX: CHOP>55 = range (mean revert), CHOP<40 = trend (pullback entries)
3. FUNDING PROXY: Use volume/taker_buy_volume ratio as funding sentiment proxy
   (high taker buy = crowded long = contrarian short signal)
4. 1d HMA(21) SLOPE: Major trend bias filter
5. 1w HMA(50): Secular trend filter for asymmetric sizing

Why this should work:
- Fisher Transform has sharper signals than RSI at extremes
- Regime filter prevents wrong strategy in wrong market
- Volume sentiment proxy captures funding rate dynamics without external data
- 4h timeframe = 25-50 trades/year target (optimal fee/trade balance)
- Asymmetric sizing: larger positions when 1d and 1w agree

Timeframe: 4h (REQUIRED)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_funding_proxy_1d1w_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.67
    Enters long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate price position within range
    highest = high_s.rolling(window=period, min_periods=period).max()
    lowest = low_s.rolling(window=period, min_periods=period).min()
    
    price_range = highest - lowest
    price_range = price_range.replace(0, 1e-10)
    
    x = 0.67 * (close_s - lowest) / price_range - 0.335
    x = np.clip(x, -0.99, 0.99)  # Prevent ln domain errors
    
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher_prev = fisher.shift(1)
    
    return fisher.values, fisher_prev.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_values = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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

def calculate_funding_proxy(taker_buy_volume, volume):
    """
    Calculate funding rate proxy using taker buy volume ratio.
    High taker buy ratio = crowded longs = contrarian short signal
    """
    volume = np.where(volume == 0, 1e-10, volume)
    taker_ratio = taker_buy_volume / volume
    taker_ratio = np.clip(taker_ratio, 0, 1)
    
    # Z-score of taker ratio over 50 periods
    taker_s = pd.Series(taker_ratio)
    taker_mean = taker_s.rolling(window=50, min_periods=50).mean().values
    taker_std = taker_s.rolling(window=50, min_periods=50).std().values
    taker_std = np.where(taker_std == 0, 1e-10, taker_std)
    
    funding_zscore = (taker_ratio - taker_mean) / taker_std
    
    return funding_zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 1w HTF indicators
    hma_1w_50 = calculate_hma(df_1w['close'].values, 50)
    hma_1w_slope = calculate_hma_slope(hma_1w_50, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    funding_zscore = calculate_funding_proxy(taker_buy_volume, volume)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1w_50_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        
        if np.isnan(funding_zscore[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 1W SECULAR TREND ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.3
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.3
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 40
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_down = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === FUNDING PROXY (contrarian) ===
        funding_extreme_long = funding_zscore[i] > 1.5  # Crowded longs = short signal
        funding_extreme_short = funding_zscore[i] < -1.5  # Crowded shorts = long signal
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ADJUSTMENT ===
        current_size = BASE_SIZE
        
        # Increase size when 1d and 1w agree (secular + intermediate trend aligned)
        if trend_1d_bullish and trend_1w_bullish:
            current_size = min(0.35, BASE_SIZE * 1.25)
        elif trend_1d_bearish and trend_1w_bearish:
            current_size = min(0.35, BASE_SIZE * 1.25)
        
        # Reduce size in uncertain regime
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        long_reasons = []
        
        # Path 1: Fisher cross up + range market (mean revert)
        if fisher_cross_up and is_range_market:
            long_score += 3
            long_reasons.append('fisher_range')
        
        # Path 2: Fisher extreme + funding crowded short (contrarian)
        if fisher_extreme_low and funding_extreme_short:
            long_score += 3
            long_reasons.append('fisher_funding')
        
        # Path 3: Trend market + 1d bullish + Fisher cross (pullback)
        if is_trend_market and trend_1d_bullish and fisher_cross_up:
            long_score += 3
            long_reasons.append('trend_pullback')
        
        # Path 4: RSI oversold + Fisher extreme (double confirmation)
        if rsi_oversold and fisher_extreme_low:
            long_score += 2
            long_reasons.append('rsi_fisher')
        
        # Path 5: Price below 1d HMA + Fisher extreme (deep pullback)
        if price_below_1d_hma and fisher_extreme_low and trend_1w_bullish:
            long_score += 2
            long_reasons.append('pullback_secular_bull')
        
        # Path 6: Simple Fisher cross + RSI confirmation
        if fisher_cross_up and rsi_oversold:
            long_score += 2
            long_reasons.append('fisher_rsi')
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 40:
            new_signal = current_size * 0.6
        elif long_score >= 2:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        short_reasons = []
        
        # Path 1: Fisher cross down + range market
        if fisher_cross_down and is_range_market:
            short_score += 3
            short_reasons.append('fisher_range')
        
        # Path 2: Fisher extreme + funding crowded long
        if fisher_extreme_high and funding_extreme_long:
            short_score += 3
            short_reasons.append('fisher_funding')
        
        # Path 3: Trend market + 1d bearish + Fisher cross
        if is_trend_market and trend_1d_bearish and fisher_cross_down:
            short_score += 3
            short_reasons.append('trend_pullback')
        
        # Path 4: RSI overbought + Fisher extreme
        if rsi_overbought and fisher_extreme_high:
            short_score += 2
            short_reasons.append('rsi_fisher')
        
        # Path 5: Price above 1d HMA + Fisher extreme (rally in bear)
        if price_above_1d_hma and fisher_extreme_high and trend_1w_bearish:
            short_score += 2
            short_reasons.append('rally_secular_bear')
        
        # Path 6: Simple Fisher cross + RSI confirmation
        if fisher_cross_down and rsi_overbought:
            short_score += 2
            short_reasons.append('fisher_rsi')
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.6
        elif short_score >= 2:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 100 bars (~17 days on 4h)
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and fisher[i] > 1.0:
                new_signal = -current_size * 0.4
            elif fisher_extreme_low:
                new_signal = current_size * 0.35
            elif fisher_extreme_high:
                new_signal = -current_size * 0.35
        
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
            # Exit long if regime switches to strong trend bearish
            if position_side > 0 and is_trend_market and trend_1d_bearish:
                regime_reversal = True
            # Exit short if regime switches to strong trend bullish
            if position_side < 0 and is_trend_market and trend_1d_bullish:
                regime_reversal = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_reversal = False
        if in_position and position_side != 0:
            # Exit long when Fisher crosses above +1.5 (overbought)
            if position_side > 0 and fisher_cross_down:
                fisher_reversal = True
            # Exit short when Fisher crosses below -1.5 (oversold)
            if position_side < 0 and fisher_cross_up:
                fisher_reversal = True
        
        if stoploss_triggered or regime_reversal or fisher_reversal:
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