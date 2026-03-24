#!/usr/bin/env python3
"""
Innocent Heikin Ashi Ethereum Strategy
Converted from TradingView Pine Script
"""

import numpy as np
import pandas as pd

name = "Innocent Heikin Ashi Ethereum Strategy"
timeframe = "5m"
leverage = 1

def calculate_heikin_ashi(df):
    """Convert standard OHLC to Heikin Ashi candles."""
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha_open = np.zeros(len(df))
    ha_open[0] = df['open'].iloc[0]
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
    ha_high = np.maximum(df['high'], np.maximum(ha_open, ha_close))
    ha_low = np.minimum(df['low'], np.minimum(ha_open, ha_close))
    return ha_open, ha_high, ha_low, ha_close

def calculate_ema(series, period):
    """Calculate Exponential Moving Average."""
    ema = np.zeros(len(series))
    multiplier = 2.0 / (period + 1.0)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_lowest_low(series, period):
    """Calculate lowest low over rolling window."""
    result = np.zeros(len(series))
    for i in range(len(series)):
        start_idx = max(0, i - period + 1)
        result[i] = np.min(series[start_idx:i+1])
    return result

def calculate_pvsra_color(volume, high, low, close, open_price):
    """
    Simplified PVSRA color calculation.
    Returns: 0=red (selling), 1=green (buying), 2=blue (strong buy), 3=violet (special)
    """
    colors = np.zeros(len(volume), dtype=int)
    avg_volume = np.mean(volume[-50:]) if len(volume) >= 50 else np.mean(volume)
    
    for i in range(len(volume)):
        body = abs(close[i] - open_price[i])
        range_hl = high[i] - low[i]
        volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1.0
        
        if close[i] > open_price[i]:
            if volume_ratio > 1.5 and body > range_hl * 0.6:
                colors[i] = 2
            elif volume_ratio > 1.0:
                colors[i] = 1
            else:
                colors[i] = 1
        else:
            if volume_ratio > 1.5 and body > range_hl * 0.6:
                colors[i] = 0
            else:
                colors[i] = 0
    return colors

def generate_signals(prices):
    """
    Generate trading signals based on Heikin Ashi + PVSRA logic.
    Returns: numpy array with 1=long, 0=flat, -1=short (this strategy is long-only)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=int)
    
    if n < 200:
        return signals
    
    ha_open, ha_high, ha_low, ha_close = calculate_heikin_ashi(prices)
    ema50 = calculate_ema(ha_close, 50)
    ema200 = calculate_ema(ha_close, 200)
    lowest_low_28 = calculate_lowest_low(ha_low, 28)
    
    pvsra_colors = calculate_pvsra_color(
        prices['volume'].values,
        ha_high,
        ha_low,
        ha_close,
        ha_open
    )
    
    COLOR_RED = 0
    COLOR_GREEN = 1
    COLOR_BLUE = 2
    COLOR_VIOLET = 3
    
    last_red_vector_below_ema50 = -1
    last_green_vector_above_ema200 = 0
    last_buy_signal_index = -1
    last_sell_signal_index = -1
    red_count_under_ema50 = 0
    
    risk_reward_ratio = 1.0
    confirmation_level = 1
    
    in_position = False
    position_entry_price = 0.0
    position_stop_loss = 0.0
    position_take_profit = 0.0
    
    for i in range(n):
        if i < 200:
            continue
        
        current_close = ha_close[i]
        current_open = ha_open[i]
        current_high = ha_high[i]
        current_low = ha_low[i]
        current_color = pvsra_colors[i]
        
        if current_color == COLOR_RED and current_close < ema50[i]:
            red_count_under_ema50 += 1
        
        if current_color == COLOR_RED and current_open < ema50[i] and current_close < ema50[i]:
            last_red_vector_below_ema50 = i
        
        if (current_color == COLOR_BLUE or current_color == COLOR_GREEN) and current_open > ema200[i] and current_open > ema50[i]:
            last_sell_signal_index = i
        
        if current_color == COLOR_GREEN and current_close > ema200[i]:
            last_green_vector_above_ema200 += 1
        
        stop_loss_price = np.nan
        take_profit_price = np.nan
        
        if last_red_vector_below_ema50 >= 0 and current_color == COLOR_GREEN:
            stop_loss_price = lowest_low_28[i]
            if not np.isnan(stop_loss_price) and stop_loss_price > 0:
                take_profit_price = current_close + (current_close - stop_loss_price) * risk_reward_ratio
        
        if in_position:
            if current_low <= position_stop_loss or current_high >= position_take_profit:
                signals[i] = 0
                in_position = False
                continue
            
            if i == last_buy_signal_index + 1 and (current_color == COLOR_RED or current_color == COLOR_VIOLET):
                signals[i] = 0
                in_position = False
                continue
            
            if last_sell_signal_index >= 0 and i == last_sell_signal_index + 1:
                if current_color == COLOR_RED and current_open > ema200[i] and current_close > ema200[i]:
                    if last_green_vector_above_ema200 >= confirmation_level:
                        signals[i] = 0
                        in_position = False
                        last_sell_signal_index = -1
                        last_green_vector_above_ema200 = 0
                        continue
        
        if not in_position:
            buy_condition = (
                last_red_vector_below_ema50 >= 0 and
                current_color == COLOR_GREEN and
                current_open > ema50[i] and
                (last_buy_signal_index < 0 or i > last_buy_signal_index)
            )
            
            if buy_condition:
                if current_close < ema200[i] and red_count_under_ema50 >= confirmation_level:
                    last_buy_signal_index = i
                    last_red_vector_below_ema50 = -1
                    red_count_under_ema50 = 0
                    signals[i] = 1
                    in_position = True
                    position_entry_price = current_close
                    position_stop_loss = stop_loss_price if not np.isnan(stop_loss_price) else current_low * 0.98
                    position_take_profit = take_profit_price if not np.isnan(take_profit_price) else current_high * 1.02
                elif current_close > ema200[i] and red_count_under_ema50 >= confirmation_level:
                    last_buy_signal_index = i
                    last_red_vector_below_ema50 = -1
                    red_count_under_ema50 = 0
                    signals[i] = 1
                    in_position = True
                    position_entry_price = current_close
                    position_stop_loss = stop_loss_price if not np.isnan(stop_loss_price) else current_low * 0.98
                    position_take_profit = take_profit_price if not np.isnan(take_profit_price) else current_high * 1.02
    
    for i in range(n):
        if signals[i] == 0 and i > 0 and signals[i-1] == 1:
            if not in_position or i > last_buy_signal_index + 50:
                pass
    
    return signals

if __name__ == "__main__":
    print(f"Strategy: {name}")
    print(f"Timeframe: {timeframe}")
    print(f"Leverage: {leverage}")
