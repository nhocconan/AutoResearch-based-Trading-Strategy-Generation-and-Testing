#!/usr/bin/env python3
"""
Experiment #117: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Pullback

Hypothesis: Previous complex regime-switching strategies failed due to overfitting and
too many conflicting filters (resulting in 0 trades). Research shows Donchian breakout
combined with HMA trend filtering worked well on SOL (Sharpe +0.782). This strategy:

1. DONCHIAN(20) BREAKOUT: Price breaks 20-day high/low for trend direction
2. HMA(21) TREND: Confirms major trend alignment (avoid counter-trend breakouts)
3. RSI(14) PULLBACK: Enter on pullbacks within trend (RSI 40-60 for longs, 40-60 for shorts)
4. 1W HMA BIAS: Weekly trend filter for major market direction
5. ATR(14) STOPLOSS: 2.5*ATR trailing stop to limit drawdown

Why this should work:
- Simpler = more trades (avoid 0-trade failure mode)
- Donchian breakouts capture major moves (BTC 2021 rally, 2022 crash)
- HMA filters false breakouts in choppy markets
- RSI pullback entries improve win rate vs chasing breakouts
- 1d timeframe = 20-50 trades/year target (low fee drag)
- 1w HTF prevents fighting secular trends

Timeframe: 1d (REQUIRED for exp #117)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_48 = calculate_hma(close, 48)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, 20)
    
    # Price position relative to Donchian
    price_vs_donchian_high = (close - donchian_low) / (donchian_high - donchian_low + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(donchian_high[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_48[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND ===
        daily_bullish = hma_1d_21[i] > hma_1d_48[i]
        daily_bearish = hma_1d_21[i] < hma_1d_48[i]
        price_above_1d_hma = close[i] > hma_1d_21[i]
        price_below_1d_hma = close[i] < hma_1d_21[i]
        
        # === DONCHIAN POSITION ===
        near_donchian_high = price_vs_donchian_high[i] > 0.85
        near_donchian_low = price_vs_donchian_high[i] < 0.15
        breakout_high = close[i] > donchian_high[i-1] if i > 0 else False
        breakout_low = close[i] < donchian_low[i-1] if i > 0 else False
        
        # === RSI CONDITIONS ===
        rsi_neutral_long = 35 < rsi_14[i] < 60
        rsi_neutral_short = 40 < rsi_14[i] < 65
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if weekly_bullish and daily_bullish:
            current_size = BASE_SIZE  # Full size in strong uptrend
        elif weekly_bearish and daily_bearish:
            current_size = BASE_SIZE  # Full size in strong downtrend
        else:
            current_size = BASE_SIZE * 0.7  # Reduced in mixed signals
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for more trades
        long_confidence = 0
        
        # Path 1: Donchian breakout + trend alignment + RSI confirmation
        if breakout_high and daily_bullish and rsi_neutral_long:
            long_confidence += 3
        
        # Path 2: Weekly bullish + daily pullback to HMA + RSI oversold
        if weekly_bullish and price_below_1d_hma and rsi_oversold:
            long_confidence += 2
        
        # Path 3: Near Donchian low + extreme RSI (mean revert in uptrend)
        if weekly_bullish and near_donchian_low and rsi_extreme_low:
            long_confidence += 2
        
        # Path 4: Simple trend follow (price > HMA21 > HMA48 + RSI > 50)
        if price_above_1d_hma and daily_bullish and rsi_14[i] > 50:
            long_confidence += 1
        
        # Path 5: Weekly bullish + price above 1w HMA (secular bull)
        if weekly_bullish and price_above_1w_hma and rsi_neutral_long:
            long_confidence += 1
        
        if long_confidence >= 3:
            new_signal = current_size
        elif long_confidence == 2 and bars_since_last_trade > 20:
            new_signal = current_size * 0.7
        elif long_confidence >= 1 and bars_since_last_trade > 40:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: Donchian breakout + trend alignment + RSI confirmation
        if breakout_low and daily_bearish and rsi_neutral_short:
            short_confidence += 3
        
        # Path 2: Weekly bearish + daily rally to HMA + RSI overbought
        if weekly_bearish and price_above_1d_hma and rsi_overbought:
            short_confidence += 2
        
        # Path 3: Near Donchian high + extreme RSI (mean revert in downtrend)
        if weekly_bearish and near_donchian_high and rsi_extreme_high:
            short_confidence += 2
        
        # Path 4: Simple trend follow (price < HMA21 < HMA48 + RSI < 50)
        if price_below_1d_hma and daily_bearish and rsi_14[i] < 50:
            short_confidence += 1
        
        # Path 5: Weekly bearish + price below 1w HMA (secular bear)
        if weekly_bearish and price_below_1w_hma and rsi_neutral_short:
            short_confidence += 1
        
        if short_confidence >= 3:
            new_signal = -current_size
        elif short_confidence == 2 and bars_since_last_trade > 20:
            new_signal = -current_size * 0.7
        elif short_confidence >= 1 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~60 days on 1d)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and daily_bullish and rsi_14[i] > 45:
                new_signal = current_size * 0.4
            elif weekly_bearish and daily_bearish and rsi_14[i] < 55:
                new_signal = -current_size * 0.4
            elif rsi_extreme_low:
                new_signal = current_size * 0.3
            elif rsi_extreme_high:
                new_signal = -current_size * 0.3
        
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
            if position_side > 0 and daily_bearish and weekly_bearish:
                trend_reversal = True
            if position_side < 0 and daily_bullish and weekly_bullish:
                trend_reversal = True
        
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